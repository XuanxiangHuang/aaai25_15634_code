# Install the require packages:
`pip install -r requirements.txt`


# To Reproduce the experiments:

### Enumerate all formal explanations and count the occurrence of relevant feature:
`python3 expFRP.py -bench dt_ijar_examples.txt dt`

### Compute SHAP scores:
`python3 expUseSHAP.py -bench dt_ijar_examples.txt dt`

### Compute sSHAP scores:
`python3 expSHAP_with_valFunc.py -bench dt_ijar_examples.txt dt`

### FRP vs. SHAP, and FRP vs. sSHAP
`python3 FRP-SHAP.py -bench dt_ijar_examples.txt`
