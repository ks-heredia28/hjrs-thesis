import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRESAVED_LINEAR = PROJECT_ROOT / "src" / "2_dashboard"


DASHBOARD_CONFIG = {
    "file_paths": {
        "llm_meta_tasks": os.path.join(PRESAVED_LINEAR, 'llm_meta_tasks_modified.xlsx'),
        "jobs_meta_tasks": os.path.join(PRESAVED_LINEAR, 'jobs_and_meta_tasks.xlsx'),
        "mock_cognitive_data": os.path.join(PRESAVED_LINEAR, 'mock_cognitive_data.xlsx'),
        "brainsfirstlogo": os.path.join(PRESAVED_LINEAR, 'brainsfirstlogo.JPG'),
        "UMAP_1_model": os.path.join(PRESAVED_LINEAR, 'UMAP_1_ridge_model.pkl'),
        "UMAP_2_model": os.path.join(PRESAVED_LINEAR, 'UMAP_2_ridge_model.pkl'),
        "UMAP_3_model": os.path.join(PRESAVED_LINEAR, 'UMAP_3_ridge_model.pkl'),
        "UMAP_1_scaler": os.path.join(PRESAVED_LINEAR, 'UMAP_1_scaler.pkl'),
        "UMAP_2_scaler": os.path.join(PRESAVED_LINEAR, 'UMAP_2_scaler.pkl'),
        "UMAP_3_scaler": os.path.join(PRESAVED_LINEAR, 'UMAP_3_scaler.pkl'),
        "llm_meta_tasks_nl" : os.path.join(PRESAVED_LINEAR, 'llm_meta_tasks_nl.xlsx'),
        "cognitive_data1": os.path.join(PRESAVED_LINEAR, 'cognitive_data_1.xlsx')
    }
}

# --- TRANSLATIONS ---
TRANSLATIONS = {
    'en': {
        "page_title": "Task Preference Dashboard",
        "welcome_id_title": "Welcome! Please Enter Your Access Code",
        "id_prompt": "Enter your Access Code to begin:",
        "id_placeholder": "e.g., 803vn0idncns0",
        "submit_id_button": "Submit Code",
        "id_not_found_error": "Access Code not found. Please check your code and try again.",
        "id_warning": "Please enter an Access Code.",
        "welcome_title": "Welcome to the Task Preference Dashboard! 👋",
        "welcome_markdown": """
            We're thrilled to have you here! Your feedback is crucial for helping us empower you and the entire BrainsFirst family to reach your maximum potential.
            
            This study involves two simple steps:
            1. Rating job-related tasks based on your personal preferences.
            2. Rating your personalized job recommendations.

            Please take your time to read the instructions and answer honestly. There are no right or wrong answers, and your unique perspective is what matters most.

            By clicking **"Start"**, you consent to BrainsFirst using your responses for product development, and sharing anonymous data with the University van Amsterdam (UvA) for a thesis research project. The UvA will not be able to identify participants, and requests for data deletion must be directly submitted to BrainsFirst. 
            """,
        "start_button": "Start",
        "task_instructions_title": "Task Rating Instructions",
        "task_instructions_markdown": """
            On the next page, you will be shown a series of job-related tasks, one at a time. Each task represents a broad category of activities common in many different jobs. Below each task title, you will find a short description along with concrete examples to help you understand what it involves.
            
            To rate each task, use the 5 buttons at the bottom of the page to indicate how much you would like or dislike doing it as part of your work:
            - **Strongly would not like doing this**: I would strongly dislike doing this task. Having this as a core part of my job would be a deal-breaker.
            - **Would not like doing this**: I would prefer to avoid this task. Doing it regularly would lower my energy and job satisfaction.
            - **Neutral**: This task doesn't strongly affect my job satisfaction either way. I may not care much about it, or I might find it interesting but not for my job.
            - **Would like doing this**: I would be happy to do this task. It's a positive part of a job, even if it's not the main reason I'm there.
            - **Strongly would like doing this**: I would actively seek out this task. Doing it would give me energy and a sense of deep satisfaction.
            
            Take your time!
            """,
        "continue_button": "Continue",
        "open_questions_header": "A Few Questions About You",
        "open_questions_info": "For this section, please answer the following questions as honestly as possible. You can choose to elaborate as much as you'd like.",
        "q1_label": "What is your current job title?",
        "q2_label": "How does your current job match you as a person?",
        "q3_label": "What job do you see yourself having in 5 years?",
        "q4_label": "How would that job match you as a person?",
        "submit_responses_button": "Submit Responses",
        "responses_saved_success": "Your responses have been saved. Thank you!",
        "spinner_text": "Analyzing your preferences and generating recommendations...",
        "recommendation_instructions_title": "Rating Your Job Recommendations",
        "recommendation_instructions_markdown": """
            On the next page, you will see several personalized job recommendations. Please rate each one based on how relevant it feels to you.

            By **relevant** we mean: Would you **enjoy** it? Could you **do** it? Does it fit your **interests** and **abilities**?

            Please rate each job **independently**, and **focus only on fit** (ignore pay, location, opening, or requirements).

            Here is an explanation of the scale:

            - *1: Highly irrelevant* = This job is completely detached from my personal skills and preferences
            - *2: Very irrelevant*
            - *3: Irrelevant*
            - *4: Somewhat irrelevant*
            - *5: Neither relevant nor irrelevant* 
            - *6: Somewhat relevant*
            - *7: Relevant*
            - *8: Very relevant*
            - *9: Highly relevant* = This job completely matches my personal skills and preferences.
            """,
        "recommendations_header": "✨ Your Job Recommendations",
        "recommendations_rating_prompt": """
            #### Please use the buttons below to rate how relevant each recommendation is to *you*, based on your personal skills and preferences.
                        
            - *1: Highly irrelevant* = This job is completely detached from my personal skills and preferences
            - *5: Neither relevant nor irrelevant* = This job aligns with some of my skills and preferences, but not all. 
            - *9: Highly relevant* = This job completely matches my personal skills and preferences.
            """,
        "progress_recommendation_text": "Recommendation {current} of {total}",
        "relevance_question": "**How relevant is this job to you?**",
        "thank_you_header": "Thank You!",
        "submit_feedback_prompt": "You have rated all your job recommendations. Click the button below to finalize and submit your feedback.",
        "submit_all_feedback_button": "Submit All Feedback",
        "feedback_submitted_success": "Thank you for submitting your feedback!",
        "restart_button": "Restart Assessment",
        "task_rating_title": "Rate Your Preferred Tasks",
        "progress_task_text": "Task {current} of {total}",
    },
    'nl': {
        "page_title": "Dashboard Taakvoorkeuren",
        "welcome_id_title": "Welkom! Voer alstublieft uw toegangscode in",
        "id_prompt": "Voer uw toegangscode in om te beginnen:",
        "id_placeholder": "bijv., 803vn0idncns0",
        "submit_id_button": "Code verzenden",
        "id_not_found_error": "Toegangscode niet gevonden. Controleer uw code en probeer het opnieuw.",
        "id_warning": "Voer alstublieft een toegangscode in.",
        "welcome_title": "Welkom bij het Dashboard Taakvoorkeuren! 👋",
        "welcome_markdown": """
            Bedankt voor uw deelname aan dit onderzoek. Uw feedback is zeer waardevol en helpt ons om onze dienstverlening voor u en andere gebruikers te verbeteren.
            
            Het onderzoek bestaat uit de volgende drie onderdelen:
            1. Het beoordelen van werkgerelateerde taken op basis van uw persoonlijke voorkeur.
            2. Het evalueren van uw persoonlijke baanaanbevelingen.
            3. Het beantwoorden van enkele vragen over uw functie.

            Neem de tijd om de instructies zorgvuldig te lezen. Het gaat om uw persoonlijke perspectief; er zijn geen goede of foute antwoorden.

            Door op **"Start"** te klikken, geeft u BrainsFirst toestemming om uw antwoorden voor onderzoeksdoeleinden te gebruiken. Uw gegevens worden volledig geanonimiseerd en verwerkt in overeenstemming met de Algemene Verordening Gegevensbescherming (AVG), zoals goedgekeurd door de Ethische Toetsingscommissie (ETC) van de Universiteit van Amsterdam.
            """,
        "start_button": "Start",
        "task_instructions_title": "Instructies voor Taakbeoordeling",
        "task_instructions_markdown": """
            Op de volgende pagina krijgt u een reeks werkgerelateerde taken te zien, één voor één. Elke taak vertegenwoordigt een brede categorie van activiteiten die in veel verschillende banen voorkomen. Onder elke taaktitel vindt u een korte beschrijving met concrete voorbeelden om u te helpen begrijpen wat het inhoudt.
            
            Om elke taak te beoordelen, gebruikt u de 5 knoppen onderaan de pagina om aan te geven in hoeverre u het leuk of niet leuk zou vinden om dit als onderdeel van uw werk te doen:
            
            - **Vind ik absoluut niet leuk om te doen**: Ik zou deze taak absoluut niet graag doen. Als dit een kernonderdeel van mijn baan zou zijn, zou dat een 'deal-breaker' zijn.
            - **Vind ik niet leuk om te doen**: Ik zou deze taak liever vermijden. Het regelmatig uitvoeren ervan zou mijn energie en werkplezier verminderen.
            - **Neutraal**: Deze taak heeft geen sterke invloed op mijn werkplezier. Het kan me niet veel schelen, of ik vind het misschien interessant, maar niet voor mijn werk.
            - **Vind ik leuk om te doen**: Ik zou deze taak graag doen. Het is een positief onderdeel van een baan, ook al is het niet de belangrijkste reden dat ik er ben.
            - **Vind ik absoluut geweldig om te doen**: Ik zou deze taak actief opzoeken. Het uitvoeren ervan zou me energie en een diep gevoel van voldoening geven.
            
            Neem de tijd!
            """,
        "continue_button": "Doorgaan",
        "open_questions_header": "Een paar vragen over u",
        "open_questions_info": "Beantwoord voor dit gedeelte de volgende vragen zo eerlijk mogelijk. U kunt ervoor kiezen om zo uitgebreid te antwoorden als u wilt.",
        "q1_label": "Wat is uw huidige functietitel?",
        "q2_label": "Hoe past uw huidige baan bij u als persoon?",
        "q3_label": "Welke baan ziet u zichzelf over 5 jaar hebben?",
        "q4_label": "Hoe zou die baan bij u als persoon passen?",
        "submit_responses_button": "Antwoorden Indienen",
        "responses_saved_success": "Uw antwoorden zijn opgeslagen. Dank u wel!",
        "spinner_text": "Uw voorkeuren worden geanalyseerd en aanbevelingen worden gegenereerd...",
        "recommendation_instructions_title": "Uw Baanaanbevelingen Beoordelen",
        "recommendation_instructions_markdown": """
            Op de volgende pagina ziet u verschillende gepersonaliseerde baanaanbevelingen. Beoordeel elke aanbeveling op basis van hoe relevant deze voor u voelt.

            Met **relevant** bedoelen we: Zou u het **leuk** vinden? Zou u het **kunnen** doen? Past het bij uw **interesses** en **vaardigheden**?

            Beoordeel elke baan **afzonderlijk** en **focus alleen op de match** (negeer salaris, locatie, vacature of vereisten).

            Hier is een uitleg van de schaal:

            - *1: Zeer irrelevant* = Deze baan staat volledig los van mijn persoonlijke vaardigheden en voorkeuren.
            - *2: Erg irrelevant*
            - *3: Irrelevant*
            - *4: Enigszins irrelevant*
            - *5: Noch relevant, noch irrelevant* 
            - *6: Enigszins relevant*
            - *7: Relevant*
            - *8: Erg relevant*
            - *9: Zeer relevant* = Deze baan komt volledig overeen met mijn persoonlijke vaardigheden en voorkeuren.
            """,
        "recommendations_header": "✨ Uw Baanaanbevelingen",
        "recommendations_rating_prompt": """
            #### Gebruik de knoppen hieronder om te beoordelen hoe relevant elke aanbeveling is voor *u*, op basis van uw persoonlijke vaardigheden en voorkeuren.
                        
            - *1: Zeer irrelevant* = Deze baan staat volledig los van mijn persoonlijke vaardigheden en voorkeuren.
            - *5: Noch relevant, noch irrelevant* = Deze baan sluit aan bij sommige van mijn vaardigheden en voorkeuren, maar niet allemaal.
            - *9: Zeer relevant* = Deze baan komt volledig overeen met mijn persoonlijke vaardigheden en voorkeuren.
            """,
        "progress_recommendation_text": "Aanbeveling {current} van {total}",
        "relevance_question": "**Hoe relevant is deze baan voor u?**",
        "thank_you_header": "Dank u wel!",
        "submit_feedback_prompt": "U heeft al uw baanaanbevelingen beoordeeld. Klik op de knop hieronder om uw feedback af te ronden en in te dienen.",
        "submit_all_feedback_button": "Alle Feedback Indienen",
        "feedback_submitted_success": "Bedankt voor het indienen van uw feedback!",
        "restart_button": "Beoordeling Herstarten",
        "task_rating_title": "Beoordeel uw Voorkeurstaken",
        "progress_task_text": "Taak {current} van {total}",
    }
}
