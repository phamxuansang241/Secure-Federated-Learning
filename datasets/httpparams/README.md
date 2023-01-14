# HttpParamsDataset

This dataset was used to evaluate anomaly detection method in my diploma thesis. Records in this dataset represents values which can be found as values of parameters in HTTP requests. Dataset contains over 3100 values and these values falls into two categories:

- benign values (19 304 items labeled as *norm*)
- anomaly values (11 763 items labeled as *anom*)

Anomaly values falls into several attack types:

- SQL Injection attacks (10 852 items labeled as *sqli*)
- Cross-Site Scripting (532 items labeled as *xss*)
- Command Injection (89 items labeled as *cmdi*)
- Path Traversal attacks (290 items labeled as *path-traversal*)

Dataset was created using several freely available sources:

- [CSIC2010 dataset](http://www.isi.csic.es/dataset/) payload values from benign requests was used
- [sqlmap](https://github.com/sqlmapproject/sqlmap) was used to generate SQL injection samples
- [xssya](https://github.com/yehia-mamdouh/XSSYA) was used to generate Cross-Site scripting samples
- [Vega Scanner](https://github.com/subgraph/Vega/wiki/Vega-Scanner) was used to generate Command injection and Path Traversal  samples
- [FuzzDB repository](https://github.com/fuzzdb-project/fuzzdb) was used to for additional Cross-Site scripting, Command injection and Path traversal samples
