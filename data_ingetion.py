from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-102#skills-measured-prior-to-july-25-2024", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()