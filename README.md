# ML_fakenews_framework 
The function fakenews has already been built and pushed and also deployed on OpenFaas as serverless function. It's in the repository fakenews_serverless. The essence of this ML framework is to be able to use the serverless function deployed on OpenFaas to training ML models.

## Steps
Step 1: Open your terminal, clone the ML_fakenews_framework repository into your local machine.

Step 2: cd into the ML_fakenews_framework directory.

Step 3: Do the necessary installations by running the commands: bash setup1.sh and bash setup2.sh

Once you have run bash setup1.sh and bash setup2.sh on your machine, you don't need to rerun them when next you start your system.

Step 3a: Whenever you reboot your system, go to your terminal and switch user to userfaas by running the command: su -- userfaas

Step 3b: Cd into the fakenews directory and run the command: bash openfaas.sh

Step 4: When the setup is complete, you will see a 25-digit password on the terminal. The 25 digit password contains both letters and numbers, copy it.

Step 5: On your browser, go to http://127.0.0.1:8080/ui/

Step 6: Type the username as admin and the password as the 25-digit password you copied from your terminal, and sign in.

Step 7: Click Deploy New Function

Step 8: Click Custom

Step 9: On the Docker Image space, type tissadeking/fakenews:latest or your_docker_user_ID/fakenews:latest

Step 10: On the Function Name space, type fakenews

Step 11: At the bottom right, click Deploy

Step 12: The fakenews function would then be deployed and after some time it would be ready to be invoked

Step 13: To perform distributed learning with the model(s) being trained on the data partitions in parallel, run the command: python3 parallel.py

Step 14: To perform distributed learning with the model(s) being trained on the data partitions one after the other, run the command: python3 sequential.py

Step 15: You can also make use of another dataset by copying the dataset into the ML_fakenews_framework directory and changing the variables: data_path, x_col and y_col at lines 47 to 49 in the parallel.py file or sequential.py file as the case may be.
