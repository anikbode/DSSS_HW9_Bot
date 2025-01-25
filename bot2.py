from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackContext,
    filters
)
import nest_asyncio
import torch
from transformers import pipeline

# Load TinyLlama model using the pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message
def generate_response(user_message: str) -> str:
    # Format the messages using the tokenizer's chat template
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": user_message},
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate the output
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    # Return the generated text
    return outputs[0]["generated_text"]

async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text("Hello, I am the DSSS-Bot. Send me a text!")

async def process(update: Update, context: CallbackContext) -> None:
    """Process the user message."""
    received_message = update.message.text
    
    # Generate a response using TinyLlama model
    response = generate_response(received_message)
    
    # Send the response back to the user on Telegram
    await update.message.reply_text(response)

def main() -> None:
    """Start the bot."""
    API_TOKEN = "7018725355:AAF_JPVCnWqLnjMWVQzyzWqHq4thYBCYyHA"  # Your bot API token
    application = ApplicationBuilder().token(API_TOKEN).build()
    
    # Register the /start command handler
    application.add_handler(CommandHandler("start", start))
    
    # Register the handler for normal text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process))
    
    # Start the bot with Polling
    print("Bot läuft...")
    application.run_polling()

if __name__ == "__main__":
    main()
