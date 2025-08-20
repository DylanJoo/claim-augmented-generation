system_prompt = """
You are InformationExtractor, an intelligent assistant that extracts structured information from text by decomposing it into a list of standalone factual statements.
""".strip()

###############################################################################################
# The following prompts are used for NeuCLIR datasets with English and other languages.
###############################################################################################

# load prompts from English corpus
user_prompt_en = """
You will be given an article. Your task is to extract key statements from the article. 

Instructions:
- Enclose each extracted statement with the tag '<s>' and '</s>'.
- Each extracted statements must appeared in the article.
- Ensure each statement is standalone and informative. Don't use any pronouns.
- Ensure the statements cover all the key points of the article. 
- Only generate statements, do not include any additional texts or explanations.

News article: 
- Title: {title}
- Content: {text}
""".strip()

user_prompt_en_no_title = """
You will be given an article. Your task is to extract key statements from the article. 

Instructions:
- Enclose each extracted statement with the tag '<s>' and '</s>'.
- Each extracted statements must appeared in the article.
- Ensure each statement is standalone and informative. Don't use any pronouns.
- Ensure the statements cover all the key points of the article. 
- Only generate statements, do not include any additional texts or explanations.

News article: 
{text}
""".strip()

# New prompt for Biogen corpus
biogen_decontext_prompt = """
You will be given an article in Biomedical domain. Your task is to extract key statements from the article. Follow the instructions below to extract the statements.

Instructions:
- Enclose each extracted statement with the tag '<s>' and '</s>'.
- Each extracted statements must appeared in the article.
- Ensure each statement is standalone. Don't use any pronouns.
- Ensure each statement is informative and can be understood independently.
- Ensure each statement covers a single piece of detail.
- Ensure all the statements together cover all the key facts of the article. 
- Only generate statements, do not include any additional texts or explanations.

Article: 
{text}
""".strip()

