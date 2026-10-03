# OS-TERM-PAPER-B-Page-Replacement-Memory-Management
# CSE-307 Track 1 --- Learned Page Replacement



## 1. Project Overview

This project implements classical page replacement algorithms and
compares them with a simple learned/adaptive page replacement policy.

The implemented policies are:

-   FIFO (First-In, First-Out)
-   LRU (Least Recently Used)
-   Optimal (Belady's)
-   Learned page replacement using a lightweight logistic regression
    classifier

The project also includes a workload-shift experiment. The first half of
the page-access trace follows a locality/sequential pattern, while the
second half changes to a random/bursty pattern.

The goal is to observe how different page replacement policies behave
when the workload changes.

## 2. Assignment Requirements Covered

This project covers the Track 1 requirements:

-   Implement FIFO, LRU, and Optimal page replacement.
-   Add a learned/adaptive component using simple page-access features.
-   Use recency and access-frequency information.
-   Generate a synthetic trace with a deliberate workload shift.
-   Measure page-fault count before and after the shift.
-   Measure hit ratio before and after the shift.
-   Compare the learned policy with the classical policies.
-   Save experimental results and charts in the `results/` folder.

## 3. Learned Component

The learned policy uses a small binary logistic regression model
implemented directly in Python.

For each page currently in memory, the model uses simple features:

1.  **Recency** --- how recently the page was accessed.
2.  **Recent frequency** --- how often the page appeared in the recent
    30 references.
3.  **Longer-window frequency** --- how often the page appeared in the
    recent 100 references.
4.  **Not-recent indicator** --- whether the page has not appeared
    recently.

The training labels are generated using the Optimal policy. This
provides a reference for which page would be a good eviction candidate
when future references are known.

The learned policy then predicts an eviction probability for each page
currently in memory and selects the candidate with the highest predicted
eviction probability.

## 4. Workload and Experimental Setup

The experiment uses:

-   **Number of frames:** 4
-   **Total references:** 600
-   **First half:** 300 references with locality/sequential behavior
-   **Second half:** 300 references with random/bursty behavior
-   **Random seed:** 2026

The workload is deliberately changed halfway through the trace so that
the policies can be compared under a changing access pattern.

## 5. Results

### Total Page Faults

  Policy      Page Faults
  --------- -------------
  FIFO                388
  LRU                 398
  Optimal             205
  Learned             285

### Before and After the Workload Shift

  -----------------------------------------------------------------------
  Policy       Before Faults   After Faults     Before Hit      After Hit
                                                     Ratio          Ratio
  ----------- -------------- -------------- -------------- --------------
  FIFO                   228            160          24.0%          46.7%

  LRU                    249            150          17.0%          50.0%

  Optimal                101            107          66.3%          64.3%

  Learned                138            147          54.0%          51.0%
  -----------------------------------------------------------------------

The learned policy produced **285 total page faults**, compared with 388
for FIFO and 398 for LRU. The Optimal policy produced 205 faults because
it has knowledge of future references and therefore provides a
theoretical reference point.

For this particular generated workload, the learned policy changed from
138 faults before the shift to 147 faults after the shift. Its hit ratio
changed from 54.0% to 51.0%. In this experiment, this was the largest
deterioration among the four policies when comparing before and after
performance.

FIFO and LRU actually had fewer faults in the second half of this
particular trace. Their hit ratios increased from 24.0% to 46.7% and
from 17.0% to 50.0%, respectively. This is specific to the generated
trace and does not mean that a random workload will always improve their
performance.

Optimal changed only slightly, from 101 to 107 faults, with its hit
ratio moving from 66.3% to 64.3%.

The learned eviction prediction accuracy was **42.3%**. This indicates
that the simple feature set and lightweight classifier were able to make
useful predictions, but there is still significant room for improvement.

## 6. Conclusion

The experiment shows that page replacement performance depends strongly
on the access pattern. Optimal achieved the lowest number of faults in
this experiment because it uses future-reference information. The
learned policy performed between the classical policies and Optimal: it
produced fewer total faults than FIFO and LRU, but more than Optimal.

The workload shift also shows why an adaptive policy can be challenging.
The features used by the learned model are based mainly on recent
history, so a sudden change in access behavior can make earlier patterns
less useful. In this experiment, the learned policy's hit ratio
decreased from 54.0% to 51.0% after the shift.

Overall, the project demonstrates the basic idea of combining a
classical Operating Systems algorithm with a lightweight learned
component and testing it under a changing workload.

## 7. Project Structure

``` text
track1_learned_page_replacement/
│
├── track1.py
├── README.md
├── requirements.txt
├── CSE307_Track1_Term_Paper_FINAL.pdf
│
└── results/
    ├── results.csv
    ├── learned_decisions.csv
    ├── summary.txt
    ├── total_page_faults.png
    ├── before_after_shift.png
    └── confidence.png
```

## 8. Requirements

-   Python 3
-   matplotlib

The current implementation uses a lightweight logistic regression
written in Python, so no scikit-learn installation is required.

## 9. How to Run on Windows

Open the project folder in VS Code.

### Step 1 --- Create a virtual environment

``` powershell
python -m venv venv
```

### Step 2 --- Activate it

``` powershell
venv\Scripts\activate
```

You should see `(venv)` at the beginning of the terminal line.

### Step 3 --- Install requirements

``` powershell
python -m pip install -r requirements.txt
```

### Step 4 --- Run the experiment

``` powershell
python track1.py
```

The program will print the total page faults, hit ratios before and
after the shift, and learned eviction accuracy.

It will also save the generated results inside:

``` text
results/
```

## 10. Expected Output

A run with the project settings used for this report produced:

``` text
Experiment complete.
Results saved in: D:\term_paper_part_B\results
FIFO    : 388
LRU     : 398
Optimal : 205
Learned : 285

Hit ratios before shift:
FIFO    : 0.240 (24.0%)
LRU     : 0.170 (17.0%)
Optimal : 0.663 (66.3%)
Learned : 0.540 (54.0%)

Hit ratios after shift:
FIFO    : 0.467 (46.7%)
LRU     : 0.500 (50.0%)
Optimal : 0.643 (64.3%)
Learned : 0.510 (51.0%)

Learned eviction accuracy: 0.423
```

Small differences can occur if the workload generation or code settings
are changed.

## 11. Results Files

### `results.csv`

Contains the numerical experiment results, including:

-   total page faults
-   before-shift page faults
-   after-shift page faults
-   before-shift hit ratios
-   after-shift hit ratios
-   learned eviction accuracy
-   confidence statistics

### `learned_decisions.csv`

Contains the learned policy's individual eviction decisions and
prediction information.

### `summary.txt`

Contains a readable summary of the experiment.

### Charts

The `results/` folder contains charts generated by the Python program
for visual presentation of the experiment.

## 12. Academic Integrity and AI Assistance

This project was developed for the CSE-307 Operating Systems term paper.
AI assistance was used to help with implementation guidance, debugging,
organization, and documentation.

The student is responsible for the submitted code, experimental results,
interpretation, and explanation of the project. The code and results
should be personally reviewed and understood before submission.

## 13. Short Demo Guide

For a 3--5 minute walkthrough:

1.  Introduce the problem: page replacement and changing workloads.

2.  Show `track1.py`.

3.  Briefly explain FIFO, LRU, and Optimal.

4.  Explain the learned policy and its four features.

5.  Show the 600-reference workload and the shift after 300 references.

6.  Run:

    ``` powershell
    python track1.py
    ```

7.  Show the page-fault and hit-ratio results.

8.  Explain that the learned policy achieved 285 total faults and 42.3%
    eviction prediction accuracy.

9.  Briefly discuss the before/after change caused by the workload
    shift.
