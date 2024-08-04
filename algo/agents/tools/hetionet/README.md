## HetioNet
For more information and resources related to HetioNet, visit the GitHub repository:
[HetioNet on GitHub](https://github.com/hetio/hetionet/tree/main/hetnet/json)

## Dependencies
```
   conda install networkx
   conda install tenacity
   pip install python-Levenshtein
```
### Getting Started
1. **Download the HetioNet Data**  
   Download the file `hetionet-v1.0.json.bz2` to the `data/` directory from the above link.

2. **Decompress the Data File**  
   Decompress `hetionet-v1.0.json.bz2` to obtain `hetionet-v1.0.json` using a suitable decompression tool or command.

### Initial Setup
- **Automated Graph Creation**  
  The first time you retrieve data from HetioNet, the system will automatically generate the graph. This process takes approximately 20 minutes.
  
- **Manual Graph Creation**  
  Alternatively, you can manually initiate the graph creation process by running `python __init__.py` in the command line.

