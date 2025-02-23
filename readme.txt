# Setup Instructions

Follow these steps to correctly set up and run the model on a university system.

NOTE: Model may need to be run on cuda servers with current number of parameters and batch size. 

## Extract the Code Directory

Ensure the `code/` directory is extracted from the provided archive:

# If provided as a .tar.gz archive
tar -xvzf code.tar.gz  

# If provided as a .zip archive
unzip code.zip         

## Navigate to the Code Directory

cd code/

## Create and Set Up a Virtual Environment

# Create a directory for the virtual environment
mkdir venv
cd venv

# Initialize a virtual environment
python3 -m venv .

# Navigate back to the code directory
cd ..

# Activate the virtual environment
source venv/bin/activate

## Install Required Dependencies

# Install all required Python packages
pip3 install -r requirements.txt

# If any dependencies fail, ensure you are using Python 3.8+ and have internet access.

## Download the Pretrained Model

To use the model trained for 70 epochs, download the checkpoint from:

[Download Model (.pth)](https://drive.google.com/file/d/1UNCgujhKBsmm_FISzjx0-yHwcGR21787/view?usp=sharing)

Once downloaded, place it inside the `checkpoints/` directory:

# Ensure the checkpoints directory exists
mkdir -p checkpoints

# Move the model file into the checkpoints directory
mv vector_field.pth checkpoints/

## Run Training

# Run the training script
python3 -m main
