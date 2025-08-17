from __future__ import annotations
import os, sys, time, random, pickle, json, re, gzip
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np, pandas as pd
from tqdm import tqdm
from openai import OpenAI, OpenAIError
import multiprocessing as mp

from Bio.SeqUtils.ProtParam import ProteinAnalysis

import torch, faiss
from transformers import AutoTokenizer, AutoModel

TRAIN_PATH = "/path/to/train_data.xlsx"
TEST_PATH  = "/path/to/test_data.xlsx"
OUT_BASE   = Path("/path/to/output/predictions.csv")

EMB_CACHE  = Path("train_faiss.index")
META_CACHE = Path("train_meta.pkl")
PHYS_CACHE = Path("phys_feat.json")
REASON_CACHE = Path("reasoning_cache.pkl.gz")

TOTAL_CHUNKS = 12
CHUNK_ID     = int(sys.argv[1]) if len(sys.argv)==2 else 1

API_KEY   = 'your_api_key_here'
BASE_URL  = "https://api.deepseek.com"
MODEL_LLM = "deepseek-reasoner"

TEMPERATURE = 0.2
NUM_SHOTS   = 8
NN_CAND     = 30
α, β, γ     = 1.0, 0.5, 0.5
SEED = 123; random.seed(SEED)

SEQ_COL, SPE_COL, MIC_COL = (
    "sequence_column_name", # D-amino acid sequence
    "species_column_name", # Target species                
    "MIC_column_name",  # Minimum Inhibitory Concentration (MIC) in µg/ml
)

def phys_feats(seq:str)->Dict[str,float]:
    pa=ProteinAnalysis(seq.replace(" ","").upper())
    return dict(net_charge=pa.charge_at_pH(7),
                pI=pa.isoelectric_point(),
                gravy=pa.gravy(),
                aromaticity=pa.aromaticity())


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
ESM="facebook/esm2_t33_650M_UR50D"; D_MODEL=1280
_tok=AutoTokenizer.from_pretrained(ESM)
_mod=AutoModel.from_pretrained(
        ESM,
        torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()
@torch.inference_mode()
def esm2_embed(seqs:list[str],bs:int=16)->np.ndarray:
    outs=[]
    for i in range(0,len(seqs),bs):
        tk=_tok([s.replace(" ","").upper() for s in seqs[i:i+bs]],
                return_tensors="pt",padding=True).to(DEVICE)
        h=_mod(**tk).last_hidden_state
        m=tk["attention_mask"].unsqueeze(-1)
        emb=((h*m).sum(1)/m.sum(1))
        emb=torch.nn.functional.normalize(emb,p=2,dim=1)
        outs.append(emb.cpu())
    return torch.cat(outs).numpy()

def make_onehot(items:list[str])->Dict[str,np.ndarray]:
    uniq=sorted(set(items)); eye=np.eye(len(uniq),dtype=np.float32)
    return {k:eye[i] for i,k in enumerate(uniq)}

def build_index(df:pd.DataFrame):
    seq_emb=esm2_embed(df[SEQ_COL].tolist(),bs=8)
    oh=make_onehot(df[SPE_COL].tolist())
    vec_species=np.stack([oh[s] for s in df[SPE_COL]],dtype=np.float32)

    phys_dict=json.loads(PHYS_CACHE.read_text()) if PHYS_CACHE.exists() else {}
    phys=[]
    for s in tqdm(df[SEQ_COL],desc="phys"):
        if s not in phys_dict: phys_dict[s]=phys_feats(s)
        phys.append([phys_dict[s][k] for k in ("net_charge","pI","gravy","aromaticity")])
    Path(PHYS_CACHE).write_text(json.dumps(phys_dict))
    vec_phys=np.asarray(phys,dtype=np.float32)

    vec_all=np.hstack([α*seq_emb,β*vec_species,γ*vec_phys]).astype("float32")
    ix=faiss.IndexFlatIP(vec_all.shape[1]); ix.add(vec_all)
    faiss.write_index(ix,str(EMB_CACHE))
    META_CACHE.write_bytes(pickle.dumps(df[[SEQ_COL,SPE_COL,"MIC_log"]].to_dict("records")))

train_df=pd.read_excel(TRAIN_PATH)
train_df["MIC_log"]=np.log2(train_df[MIC_COL].astype(float)).round(0).astype(int)
if not EMB_CACHE.exists(): build_index(train_df)
faiss_index=faiss.read_index(str(EMB_CACHE))
meta=pickle.loads(META_CACHE.read_bytes())

def build_query_vec(seq:str,spe:str)->np.ndarray:
    v_seq=esm2_embed([seq])[0]
    oh=make_onehot(train_df[SPE_COL].tolist())
    v_spec=oh.get(spe,np.zeros(len(next(iter(oh.values())))))
    v_phys=np.array([phys_feats(seq)[k] for k in ("net_charge","pI","gravy","aromaticity")],dtype=np.float32)
    return np.hstack([α*v_seq,β*v_spec,γ*v_phys]).astype("float32")

def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def mmr(q,vecs,k,lamb):
    S,out=set(),[]
    for _ in range(k):
        best,best_sc=None,-1e9
        for j,v in enumerate(vecs):
            if j in S: continue
            sc=lamb*cosine(q,v)-(1-lamb)*max((cosine(v,vecs[s]) for s in S),default=0)
            if sc>best_sc: best,best_sc=j,sc
        S.add(best); out.append(best)
    return out
def retrieve_examples(seq:str,spe:str):
    q=build_query_vec(seq,spe)
    _,I=faiss_index.search(q[None,:],NN_CAND)
    ids=I[0].astype(np.int64)
    vecs=faiss_index.reconstruct_batch(ids)
    picks=mmr(q,vecs,NUM_SHOTS,0.5)
    return [meta[ids[i]] for i in picks]

client=OpenAI(api_key=API_KEY,base_url=BASE_URL)
SYS_A={"role":"system",
       "content":(
            "You are an expert in antimicrobial peptides (AMPs). "
            "Given a D-amino-acid peptide sequence (lower-case) and a bacterial species, "
            "estimate the Minimum Inhibitory Concentration (MIC, µg/ml).\n"
            "Think out loud in a few concise bullet points that cover: "
            "• key residue classes & motifs • net charge & hydrophobicity • secondary-structure tendency "
            "• how the bacterial envelope affects uptake • comparison to known analogues. "
            "Then state your MIC estimate and convert it to log₂(MIC).\n\n"
            "After your reasoning, output exactly:\n"
            "### Answer\n"
            "{\"log2_MIC\": <number>}\n"
            "The JSON object must be on a single line."
                  )}
JSON_RE=re.compile(r"\{.*\}")

def load_cache()->dict:
    if REASON_CACHE.exists():
        with gzip.open(REASON_CACHE,"rb") as f: return pickle.load(f)
    return {}
reason_cache=load_cache()
def get_reasoning(ex:dict)->str:
    key=f"{ex[SEQ_COL]}|{ex[SPE_COL]}"
    if key in reason_cache: return reason_cache[key]
    user={"role":"user","content":
          f"For the peptide sequence:\n{ex[SEQ_COL]}\n"
          f"and the target species:\n{ex[SPE_COL]},\n"
          f"the known log₂(MIC) is {int(ex['MIC_log'])}. Explain your reasoning and output JSON."}
    rsp=client.chat.completions.create(model=MODEL_LLM,
           messages=[SYS_A,user],temperature=0.1,timeout=600)
    text=rsp.choices[0].message.content.strip()
    reason_cache[key]=text
    with gzip.open(REASON_CACHE,"wb") as f: pickle.dump(reason_cache,f)
    return text

SYS_B={"role":"system",
       "content":("You are an AMP expert. Given a peptide and species, "
                  "estimate MIC. Provide reasoning then exactly:\n"
                  "### Answer\n{\"log2_MIC\": <int>}")}
def build_prompt(seq:str,spe:str,examples:List[dict])->List[dict]:
    msgs=[SYS_B]
    for ex in examples:
        msgs.append({"role":"user","content":
            f"For the peptide sequence:\n{ex[SEQ_COL]}\n"
            f"and the target species:\n{ex[SPE_COL]},\n"
            f"predict log₂(MIC)."})
        msgs.append({"role":"assistant","content":get_reasoning(ex)})
    msgs.append({"role":"user","content":
        f"For the peptide sequence:\n{seq}\n"
        f"and the target species:\n{spe},\n"
        f"predict log₂(MIC)."})
    return msgs

def parse_log2(text:str)->int:
    m=JSON_RE.search(text);  obj=json.loads(m.group()) if m else {"log2_MIC":0}
    return int(obj["log2_MIC"])

def predict(seq:str,spe:str)->int:
    examples=retrieve_examples(seq,spe)
    rsp=client.chat.completions.create(
        model=MODEL_LLM,
        messages=build_prompt(seq,spe,examples),
        temperature=TEMPERATURE,timeout=1800)
    return parse_log2(rsp.choices[0].message.content)

def main():
    df=pd.read_excel(TEST_PATH)
    df["MIC_true"]=np.log2(df[MIC_COL].astype(float)).round(0).astype(int)
    chunks=np.array_split(df,TOTAL_CHUNKS)
    chunk=chunks[CHUNK_ID-1].reset_index(drop=True)
    print(f"▶ Chunk {CHUNK_ID}/{TOTAL_CHUNKS}  ({len(chunk)} samples)")

    out=OUT_BASE.parent/f"{OUT_BASE.stem}_{CHUNK_ID}{OUT_BASE.suffix}"
    if not out.exists():
        out.write_text(f"{SEQ_COL},{SPE_COL},MIC_pred,MIC_true\n",encoding="utf-8")

    for i,row in chunk.iterrows():
        seq,spe,y=row[SEQ_COL],row[SPE_COL],row["MIC_true"]
        print(f"[{i+1}/{len(chunk)}] {seq[:25]} …")
        y_pred=predict(seq,spe)
        with out.open("a",encoding="utf-8") as f:
            f.write(f"{seq},{spe},{y_pred},{y}\n")
    print(f"✅ Finish → {out}")

if __name__=="__main__":
    mp.set_start_method('spawn',force=True)
    main()
