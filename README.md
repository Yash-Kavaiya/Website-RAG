
# Website RAG Chatbot
This project implements a Retrieval-Augmented Generation (RAG) system using Langchain, Google's Generative AI, and Gradio. It creates a chatbot interface that answers questions based on information retrieved from a pre-processed set of web pages.

## Features

- RAG system powered by Langchain and Google's Generative AI
- Local FAISS vector store for efficient information retrieval
- Customizable UI with CSS styling

## Prerequisites

- Python 3.9 or higher
- Google API key for access to Generative AI models
- Langchain API key
- Pre-processed FAISS index (stored in `./Vector_DB/faiss_index`)

## Installation

1. Clone this repository:
   ```
   git clone 
   cd website-rag-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## Usage

To run the chatbot locally:

```
python app.py
```

This will start the Gradio interface, typically accessible at `http://localhost:7860`.

## Deployment

This project is configured for deployment to Google Cloud Run. 

1. Build the Docker image:
   ```
   docker build -t website-rag-chatbot .
   ```

2. Tag and push the image to Google Container Registry:
   ```
   docker tag website-rag-chatbot gcr.io/your-project-id/website-rag-chatbot
   docker push gcr.io/your-project-id/website-rag-chatbot
   ```

3. Deploy to Cloud Run:
   ```
   gcloud run deploy website-rag-chatbot --image gcr.io/your-project-id/website-rag-chatbot --platform managed
   ```

   Remember to set the necessary environment variables in the Cloud Run configuration.

## Project Structure

- `app.py`: Main application file containing the Gradio interface and RAG logic
- `Dockerfile`: Configuration for containerizing the application
- `requirements.txt`: List of Python package dependencies
- `Vector_DB/faiss_index`: Pre-processed FAISS index (not included in repository)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Langchain for the RAG framework
- Google for the Generative AI models
- Gradio for the easy-to-use interface building

## Contact

Created by Yash Kavaiya - feel free to contact me!
- GitHub: [https://github.com/Yash-Kavaiya](https://github.com/Yash-Kavaiya)
- LinkedIn: [https://linkedin.com/in/Yash-Kavaiya](https://linkedin.com/in/Yash-Kavaiya)
