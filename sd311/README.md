# San Diego 311 - Get It Done



This folder contains the materials for a project I did using data from the City of San Diego's [Get It Done](https://www.sandiego.gov/get-it-done) application. Get It Done allows San Diego residents to perform many city related tasks like apply for parking permits and request street sweeping. The analysis done here focuses on complaints such as graffiti and cars parked for too long in one spot. The data used is a subset of the complaints from 2018 and was obtained from the [San Diego Open Data Portal](https://data.sandiego.gov).

For this project, I used the text description provided by the resident to predict the type of complaint (e.g. graffiti). The main purpose of this project was to familiarize myself with text processing. To that end, I learned about transforming raw text into numeric vectors with the [quanteda](https://quanteda.io) R package and modeling the transformed data with neural networks in [Keras](https://keras.io). Along the way I learned about the cloud computing service [Paperspace](https://www.paperspace.com) which I used to conduct the analysis.

The file containing the code I used to run the final analysis is `get-it-done.R`. I also wrote a summary, `pres.html`, to present the work to my class.

