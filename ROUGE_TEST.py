# TODO : Require Testing with a Capable System
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

# Try to initialize CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_vectorstore():
    """Initialize ChromeDB."""
    vectorstore = Chroma()
    vectorstore.delete_collection()
    return vectorstore


async def load_pdf(file_path):
    """Load the PDF"""
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def load_model(model_path):
    """Load the Model"""
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def process_text(pages, tokenizer, max_tokens=2000):
    """Truncating"""
    context = "\n".join([page.page_content for page in pages])
    tokens = tokenizer.encode(context, add_special_tokens=False)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def generate_answer(model, tokenizer, context, question):
    """Generating"""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=4000)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def ROUGE(reference, gpt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, gpt)
    return scores


def main():
    vectorstore = initialize_vectorstore()
    pages = load_pdf("nsbm_foc_data.pdf")
    model, tokenizer = load_model("cognitivecomputations/Dolphin3.0-Llama3.2-3B")
    context = process_text(pages, tokenizer)

    question = "Enter your question here"

    questionArray = ["Who is the Dean of NSBM FOC ?", "How to contact NSBM for new Enrollments ?", "What is the dean's message about ? Answer in a paragraph"]
    referenceAnswerArray = [
        "Dean of the NSBM's Faculty of computing is Dr.Rasika Ranaweera",
        "Send an Email to inquiries@nsbm.ac.lk",
        """The Dean's message welcomes students to the Faculty of Computing at NSBM Green University, emphasizing the institution’s commitment to providing innovative and industry-relevant education. It highlights the well-qualified faculty and strong student support services aimed at preparing students for successful careers. The message encourages students to work hard, set high standards, and take responsibility for their learning by effectively managing their time and utilizing available resources. Ultimately, it expresses the university’s goal of developing well-rounded individuals who will succeed in life and serve as ambassadors for NSBM."""]
    gptAnswerArray = []

    for question in questionArray:
        gptAnswer = generate_answer(model, tokenizer, context, question)
        gptAnswerArray.append(gptAnswer)

    for i in range(len(questionArray)):
        print("Calculating ROUGE Scores for Question 1")
        reference = referenceAnswerArray[i]
        gpt = gptAnswerArray[i]
        print("Reference: ", reference)
        print("GPT: ", gpt)
        scores = ROUGE(reference, gpt)
        print(f"Scores:{scores}")
        print("-----------------------------")

if __name__ == "__main__":
    main()
