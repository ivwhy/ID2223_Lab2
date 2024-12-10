# ID2223_Lab2

## Task 1
Fine-Tune a pre-trained large language (transformer) model and build a
serverless UI for using that model
● Tasks
a. Fine-tune an existing pre-trained large language model on the FineTome
Instruction Dataset
b. Build and run an inference pipeline with a Gradio UI on Hugging Face
Spaces for your model.

## Task 2
Describe in your README.md program ways in which you can improve
model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the
fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to
train a better model that one provided in the blog post
If you can show results of improvement, then you get the top grade.
2. Try out fine-tuning a couple of different open-source foundation LLMs to
get one that works best with your UI for inference (inference will be on
CPUs, so big models will be slow).
3. You are free to use other fine-tuning frameworks, such as Axolotl of HF
FineTuning - you do not have to use the provided unsloth notebook.

# ImproModel-Centric Approach

### Model-Centric Approach

- **Training Parameters:** Getting the training settings right is key to achieving the best results. Things like warmup steps and learning rates can make a big difference. Ideally, training should be done slowly for better performance, but we couldn’t fully explore this due to time constraints.  
- **Choosing the Right Model:** We avoided models using bnb-4bit since they don’t work well with many deployment platforms. However, because this project focuses on efficient 4-bit training, we chose a compatible model for our needs.  
- **Pre-trained Models:** Pre-trained instruction models aren’t always the best choice, especially for short training sessions. The improvements might not justify the effort, depending on the purpose.  

### Data-Centric Approach

- **Picking the Right Dataset:** The dataset you choose can make or break your fine-tuning process. For example, we used the FineTome-100k dataset, which is great for educational content. If your goal is different, another dataset might work better.  
- **Splitting the Data:** We split the data into training, evaluation, and testing sets to properly measure performance. Using datasets that already come pre-split can save time. The evaluation and test sets helped us track how well the model was doing.

