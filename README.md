
#### Project Summary
This project involves the analysis of anonymized and hashed fiscal data provided by an investment firm for a number of companies. The task was to predict the financial returns of these companies by projecting them into the future. However, there was a misunderstanding in the task, leading to the creation of a binary classification model instead of a regression model.

#### Classification Task
The classification model was designed to categorize companies based on their projected returns. If a company was projected to have a return of more than a certain threshold (denoted as "x"), it was classified as a "unicorn" company.

#### Approach
- Data: The dataset consisted of anonymized financial data for multiple companies.
- Model: A binary classification model was developed to predict whether a company's return would exceed the threshold "x" or not.
- Evaluation: The model's performance was evaluated based on standard classification metrics such as accuracy, precision, recall, and F1-score.
- Misunderstanding: The original regression task was misinterpreted, leading to the classification approach.

#### Purpose
Despite the deviation from the initial regression task, the classification model provides insights into identifying companies with potentially high returns, aiding in investment decision-making.

#### Future Steps
- Clarification: Ensure better understanding of task requirements in future projects.
- Refinement: Refine the classification model based on feedback and further data analysis.
- Collaboration: Collaborate with domain experts to enhance model accuracy and relevance.

### trial.py is the main file that needs to be run.
### model.py is our training model

#### how to run the file?
#### install streamlit and then set your working directory to be the same as the trial.py file, then in the command prompt type "streamlit run trial.py"
