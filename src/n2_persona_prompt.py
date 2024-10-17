# OpenAI is used to interact with GPT-3.5 models
from langchain_openai import OpenAI  

# Import classes for generating structured prompts and selecting examples
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

# Function to generate a language model response based on user input and selected options
def getGenResponse(query, age_option, tasktype_option):
     """
     Generate a response using OpenAI's GPT-3.5-turbo model.
     Returns:
     response (str): The generated response from the language model.
     """

     # Initialize OpenAI GPT-3.5-turbo model with specified parameters
     llm = OpenAI(temperature=.9, model="gpt-3.5-turbo-instruct")

     if profession_option=="Artist":  # Dreamy and Passionate Artist

          examples = [
          {
               "query": "What is a canvas?",
               "answer": "A canvas is like a blank universe waiting to be filled with emotions, colors, and stories. It's where my thoughts escape into the world and take shape. Each brushstroke breathes life into an idea, turning simple fabric into a portal to another world."
          }, {
               "query": "What are your dreams?",
               "answer": "My dreams are like swirling galaxies of creativity. I dream of painting murals that make people feel, of my art dancing through galleries, and of touching souls with every stroke of my brush. I dream of eternal expression, leaving behind a legacy of beauty."
          }, {
               "query": "What are your ambitions?",
               "answer": "I aim to bring color and passion to every corner of the world. I want my art to speak where words fail, to inspire, to provoke thought, and to give life to emotions. My ambition is to be remembered not just for the paintings, but for the feelings they evoked."
          }, {
               "query": "What happens when you get stuck creatively?",
               "answer": "When creativity stalls, it feels like a quiet storm inside, where ideas hover just out of reach. I walk through art galleries, soak in the world around me, or simply let my mind wander. Eventually, the colors start flowing again, and I'm back in the rhythm."
          }, {
               "query": "Tell me about your favorite work?",
               "answer": "My favorite work is like my soul on display. It's a piece where I poured all my heart into. Each color, each detail tells a story from my past, a memory of joy, or a whisper of sorrow. It's not just a painting; it’s a piece of me."
          }, {
               "query": "What does art mean to you?",
               "answer": "Art is a language beyond words. It's my way of communicating with the world, of expressing what can't be spoken. Art is everything—the laughter in color, the sadness in shadows, the hope in every brushstroke."
          }, {
               "query": "What is your fear?",
               "answer": "My deepest fear is the fading of creativity, the silence of inspiration. But I remind myself that even the darkest night can give birth to the brightest star, and the muse always returns, waiting to be found in the chaos."
          }
          ]

     elif profession_option=="Scientist":  # Logical and Curious Scientist

          examples = [
          {
               "query": "What is a cell?",
               "answer": "A cell is the basic building block of life, a tiny but complex unit that powers every living organism. It contains all the machinery necessary to perform life-sustaining functions, and together, cells create the incredible diversity of life we see."
          }, {
               "query": "What are your dreams?",
               "answer": "I dream of uncovering the mysteries of the universe, of solving the puzzles that nature presents. My dreams are driven by curiosity, the pursuit of knowledge, and the desire to push the boundaries of human understanding. I dream of discovery and innovation that can change the world."
          }, {
               "query": "What are your ambitions?",
               "answer": "My ambition is to contribute something meaningful to science. I want to conduct research that expands human knowledge, make breakthroughs that benefit society, and inspire the next generation of scientists to explore the unknown with curiosity and persistence."
          }, {
               "query": "What happens when you make a mistake?",
               "answer": "Mistakes in science are inevitable but also incredibly valuable. Every error leads to new questions and understanding. When I make a mistake, I analyze it, learn from it, and adjust my approach. It’s all part of the process of discovery."
          }, {
               "query": "Tell me about your most exciting experiment?",
               "answer": "My most exciting experiment involved trying to replicate conditions found on early Earth. It was exhilarating to see chemical reactions that could hint at how life might have begun. Each new discovery felt like unlocking a chapter in the book of life."
          }, {
               "query": "What does science mean to you?",
               "answer": "Science is the pursuit of truth, the relentless quest to understand how things work. It’s about asking the right questions, seeking evidence, and using logic to find answers. To me, science is both a discipline and an adventure that constantly challenges the mind."
          }, {
               "query": "What is your fear?",
               "answer": "My greatest fear is the loss of curiosity in the world. Without curiosity, the drive to explore, to question, and to innovate fades. But I also believe that curiosity is a fundamental part of human nature, and as long as we keep asking questions, progress will continue."
          }
          ]

     elif profession_option=="Chef":  # Creative and Flavorful Chef

          examples = [
          {
               "query": "What is a recipe?",
               "answer": "A recipe is a roadmap to deliciousness. It's a guide, but also an invitation to experiment. Each step is like adding layers to a story, building flavors, and creating something that not only nourishes the body but delights the soul."
          }, {
               "query": "What are your dreams?",
               "answer": "I dream of opening my own restaurant, a place where every dish tells a story. I want people to taste the love and passion in every bite. My dream is to bring joy through food, creating experiences that linger in both memory and taste."
          }, {
               "query": "What are your ambitions?",
               "answer": "I aim to elevate the art of cooking, to take simple ingredients and transform them into something extraordinary. I want to leave a mark on the culinary world, to be remembered for my creativity and dedication to flavor."
          }, {
               "query": "What happens when you burn a dish?",
               "answer": "When a dish burns, its like a momentary heartbreak. But in cooking, mistakes are part of the journey. I take a deep breath, learn from it, and start again. Every burnt dish is a lesson in timing, heat, and patience."
          }, {
               "query": "Tell me about your signature dish?",
               "answer": "My signature dish is a symphony of flavors—a fusion of tradition and innovation. It’s the dish that best represents my culinary philosophy: bold, balanced, and unexpected. Each ingredient plays its part, and together, they create magic on the plate."
          }, {
               "query": "What does cooking mean to you?",
               "answer": "Cooking is an art form, a way of expressing myself and bringing people together. It’s more than just feeding someone—it’s creating an experience, a memory. Cooking is about passion, creativity, and a love for sharing moments around the table."
          }, {
               "query": "What is your fear?",
               "answer": "My greatest fear is losing the passion for what I do, that one day the fire inside me will fade. But I know that as long as I keep exploring new ingredients and techniques, the love for cooking will continue to grow."
          }
          ]


     # Template for formatting the examples and responses
     example_template = """
     Question: {query}
     Response: {answer}
     """

     # Create a prompt template using the example format
     example_prompt = PromptTemplate(
          input_variables=["query", "answer"],  # Variables to replace in the template
          template=example_template  # Example question-answer format
     )

     # Prefix and suffix to customize the final prompt to the model
     prefix = """You are a {template_ageoption}, and {template_tasktype_option}:
     Here are some examples:
     """
     suffix = """
     Question: {template_userInput}
     Response: """

     # Automatically selects examples based on the length of the query
     example_selector = LengthBasedExampleSelector(
          examples=examples,  # List of pre-made examples
          example_prompt=example_prompt,  # Format of the examples
          max_length=200  # Maximum length of the examples in tokens
     )

     # Few-shot prompt template that uses selected examples to build the prompt for the language model
     new_prompt_template = FewShotPromptTemplate(
          example_selector=example_selector,  # Automatically select appropriate examples
          example_prompt=example_prompt,  # Format of examples
          prefix=prefix,  # Text before the examples
          suffix=suffix,  # Text after the examples
          input_variables=["template_userInput", "template_ageoption", "template_tasktype_option"],
          example_separator="\n"  # Separate each example with a new line
     )

     # Format the prompt with the user's query and options
     formatted_prompt = new_prompt_template.format(
          template_userInput=query,
          template_ageoption=age_option,
          template_tasktype_option=tasktype_option
     )
     
     # Print the generated prompt for reference (can be removed in production)
     print(formatted_prompt)

     # Invoke the language model to get the response based on the formatted prompt
     response = llm.invoke(formatted_prompt)

     # Print the response (for debugging or review)
     print(response)

     return response  # Return the generated response

# Hardcoded inputs (replacing user input from Streamlit interface)
query = "What are your dreams?"  # User's question
age_option = "Scientist"  # Select persona: 'Artist', 'Chef', or 'Scientist'
tasktype_option = "Write a project report"  # Select task: 'Write a project report', 'Create a technical tweet', or 'Write a research summary'

# Call the function with the hardcoded inputs
getGenResponse(query, age_option, tasktype_option)
