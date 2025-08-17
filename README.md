# PeptiD-Agent

PeptiD-Agent is a peptide activity prediction framework powered by DeepSeek LLMs.  

---

## 1. Environment Setup

We recommend using **Anaconda** to manage the environment.  
Please create the environment directly from our provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate peptid-agent
```

---

## 2. API Key Configuration

PeptiD-Agent relies on **DeepSeek LLMs**.  
You must provide your own API key:

1. Open `PeptiD-Agent.py`  
2. Go to **line 28** and replace the placeholder with your key:

```python
API_KEY = 'your_api_key_here'
```

---

## 3. Data Splitting with `TOTAL_CHUNKS`

```python
TOTAL_CHUNKS = 12
```

This parameter controls how the test data is divided into equal parts:  

- Allows running multiple processes on different machines/terminals in parallel  
- Reduces potential losses due to unstable network connections  

For example, if `TOTAL_CHUNKS = 12`, the test data will be split into 12 equal subsets.

---

## 4. Usage

Run the script with a chunk index as input:

```bash
python PeptiD-Agent.py 1
```

Here, `1` indicates which data subset (out of `TOTAL_CHUNKS`) you want to process.

---

## 5. Dataset Path & Column Settings

In the script, modify the dataset paths and column names according to your own data:

```python
TRAIN_PATH = 'path/to/your/training_data.csv'
TEST_PATH  = 'path/to/your/test_data.csv'

SEQ_COL = 'sequence_column_name'
SPE_COL = 'species_column_name'
MIC_COL = 'MIC_column_name'
```

- `SEQ_COL`: column name for peptide sequences  
- `SPE_COL`: column name for species  
- `MIC_COL`: column name for Minimum Inhibitory Concentration (MIC)  

Make sure these correspond to your dataset schema.

---

## Citation

If you use **PeptiD-Agent** in your research, please cite our work.  
(Citation format can be added here once the paper is available.)
