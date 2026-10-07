from datetime import date, datetime
import re

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Tools
from travel_agent_api.tools.flights_finder import flights_finder
from travel_agent_api.tools.hotels_finder import hotels_finder
from travel_agent_api.tools.chain_historical_expert import chain_historical_expert
from travel_agent_api.tools.chain_travel_plan import chain_travel_plan


FLIGHTS_OUTPUT = """
format: markdown

## Miglior Opzione

### Andata:
- Compagnia aerea: Ryanair
- Data di partenza: 2024-12-13
- Ora di partenza: 10:00
- Durata del volo: 1h 30m

### Ritorno:
- Compagnia aerea: Ryanair
- Data di ritorno: 2024-12-19
- Ora di partenza: 14:30
- Durata del volo: 1h 30m

Inserisci il link di Google per la prenotazione se possibile.

#### Altre opzioni disponibili:

- Compagnia aerea: Ryanair
- Data di partenza: 2024-12-13
- Ora di partenza: 10:00
- Durata del volo: 1h 30m

- Compagnia aerea: Ryanair
- Data di ritorno: 2024-12-19
- Ora di partenza: 14:30
- Durata del volo: 1h 30m

...
"""


HOTELS_OUTPUT = """
format: markdown

#### Hotel 1

Inserisci la foto dell'hotel se disponibile.

*Descrizione:* Camere e suite eleganti, a volte con vista sulla città, in hotel esclusivo con piscina panoramica e spa.

*Prezzo per notte:* €296 (prima delle tasse e spese: €260)

*Prezzo totale:* €2,660 (prima delle tasse e spese: €2,336)

*Check-in:* 15:00, Check-out: 12:00

*Valutazione complessiva:* 4.5 su 5

*Servizi Inclusi:* Spa, Piscina, Parcheggio gratuito


#### Hotel 2

Inserisci la foto dell'hotel se disponibile.

*Descrizione:* Hotel in stile Liberty con alloggi arredati in maniera artistica, ristorante elegante, bar e spa.

*Prezzo per notte:* €380 (prima delle tasse e spese: €333)

*Prezzo totale:* €3,418 (prima delle tasse e spese: €3,000)

*Check-in:* 15:00, Check-out: 12:00

*Valutazione complessiva:* 4.5 su 5

*Servizi Inclusi:* Spa, Piscina, Parcheggio gratuito
"""


TRAVEL_PLAN_OUTPUT = """
format: markdown

### Itinerario:

### Giorno 1 - 2024-12-13:

*Mattina:* Descrizione dell'attivita' da svolgere la mattina

*Pomeriggio:* Descrizione dell'attivita' da svolgere il pomeriggio

*Sera:* Descrizione dell'attivita' da svolgere la sera


### Giorno 2 - 2024-12-14:

*Mattina:* Descrizione dell'attivita' da svolgere la mattina

*Pomeriggio:* Descrizione dell'attivita' da svolgere il pomeriggio

*Sera:* Descrizione dell'attivita' da svolgere la sera

...
"""


class Agent:

    def __init__(self):

        self.model = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        )

        self.tools = [
            chain_historical_expert,
            flights_finder,
            hotels_finder,
            chain_travel_plan,
        ]

    # ========================================================
    # RECUPERO MESSAGGI UTENTE
    # ========================================================

    def get_user_messages(self, messages: list) -> list:
        """
        Recupera tutti i messaggi dell'utente dalla conversazione.

        Supporta sia i messaggi in formato dict sia gli oggetti
        HumanMessage di LangChain.
        """

        user_messages = []

        for message in messages:

            # ------------------------------------------------
            # Messaggio in formato dict
            # ------------------------------------------------

            if isinstance(message, dict):

                if message.get("role") != "user":
                    continue

                content = message.get("content", "")

                if isinstance(content, str):
                    user_messages.append(content)

                continue

            # ------------------------------------------------
            # Messaggio LangChain
            # ------------------------------------------------

            message_type = getattr(
                message,
                "type",
                None
            )

            if message_type != "human":
                continue

            content = getattr(
                message,
                "content",
                ""
            )

            if isinstance(content, str):
                user_messages.append(content)

        return user_messages

    # ========================================================
    # CONTROLLO DATE
    # ========================================================

    def validate_dates(
        self,
        messages: list,
        request_type: str
    ) -> str | None:
        """
        Verifica che siano presenti le date necessarie per
        la richiesta e che non siano nel passato.

        Regole:

        FLIGHT:
        - almeno una data di partenza obbligatoria.

        COMPLETE_TRIP:
        - data di partenza obbligatoria;
        - data di ritorno/fine viaggio obbligatoria.

        ITINERARY:
        - data di partenza obbligatoria;
        - data di fine viaggio obbligatoria.

        HOTEL:
        - check-in e check-out obbligatori.

        Le date devono essere espresse nel formato:

            YYYY-MM-DD

        """

        user_messages = self.get_user_messages(messages)

        if not user_messages:
            return None

        conversation = " ".join(user_messages)

        # ----------------------------------------------------
        # CERCA LE DATE NELLA CONVERSAZIONE
        # ----------------------------------------------------

        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"

        date_strings = re.findall(
            date_pattern,
            conversation
        )

        valid_dates = []

        for date_string in date_strings:

            try:

                parsed_date = datetime.strptime(
                    date_string,
                    "%Y-%m-%d"
                ).date()

                valid_dates.append(parsed_date)

            except ValueError:

                continue

        # ----------------------------------------------------
        # CONTROLLO FORMATO DATE
        # ----------------------------------------------------

        # Se l'utente ha inserito date come 15/11/2026,
        # il sistema non le usa automaticamente perché i tool
        # richiedono YYYY-MM-DD.
        european_date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"

        european_dates = re.findall(
            european_date_pattern,
            conversation
        )

        if european_dates and not valid_dates:

            return (
                "📅 Per poter cercare i voli ho bisogno delle "
                "date nel formato **YYYY-MM-DD**.\n\n"
                "Ad esempio: **2026-11-15**."
            )

        # ----------------------------------------------------
        # CONTROLLO DATE NEL PASSATO
        # ----------------------------------------------------

        today = date.today()

        for parsed_date in valid_dates:

            if parsed_date < today:

                return (
                    "⚠️ Le date inserite non sono valide.\n\n"
                    f"La data **{parsed_date.strftime('%Y-%m-%d')}** "
                    f"è già passata. Oggi è "
                    f"**{today.strftime('%Y-%m-%d')}**.\n\n"
                    "Inserisci una data futura per poter "
                    "organizzare il viaggio."
                )

        # ----------------------------------------------------
        # NESSUNA DATA
        # ----------------------------------------------------

        if request_type == "flight":

            if len(valid_dates) == 0:

                return (
                    "📅 Per poter cercare i voli mi serve sapere "
                    "la **data di partenza**.\n\n"
                    "Indicamela nel formato **YYYY-MM-DD**.\n\n"
                    "Ad esempio: **2026-11-15**."
                )

        # ----------------------------------------------------
        # VIAGGIO COMPLETO
        # ----------------------------------------------------

        if request_type == "complete_trip":

            if len(valid_dates) == 0:

                return (
                    "📅 Prima di iniziare la ricerca mi servono "
                    "le **date del viaggio**.\n\n"
                    "Indicami la data di partenza e la data di "
                    "ritorno nel formato **YYYY-MM-DD**.\n\n"
                    "Ad esempio: **2026-11-15 → 2026-11-20**."
                )

            if len(valid_dates) == 1:

                return (
                    "📅 Ho la data di partenza, ma mi serve anche "
                    "la **data di ritorno**.\n\n"
                    "Indicami entrambe le date nel formato "
                    "**YYYY-MM-DD**."
                )

        # ----------------------------------------------------
        # ITINERARIO
        # ----------------------------------------------------

        if request_type == "itinerary":

            if len(valid_dates) == 0:

                return (
                    "📅 Per creare l'itinerario mi serve sapere "
                    "la **data di partenza** del viaggio."
                )

            if len(valid_dates) == 1:

                return (
                    "📅 Ho la data di partenza, ma mi serve anche "
                    "la **data di fine del viaggio**."
                )

        # ----------------------------------------------------
        # HOTEL
        # ----------------------------------------------------

        if request_type == "hotel":

            if len(valid_dates) == 0:

                return (
                    "📅 Per cercare l'hotel mi servono le date "
                    "di **check-in e check-out**."
                )

            if len(valid_dates) == 1:

                return (
                    "📅 Ho una delle due date, ma mi serve anche "
                    "la seconda data per poter cercare l'hotel.\n\n"
                    "Indicami il **check-in e il check-out**."
                )

        # ----------------------------------------------------
        # CONTROLLO ORDINE DELLE DATE
        # ----------------------------------------------------

        if len(valid_dates) >= 2:

            start_date = valid_dates[0]
            end_date = valid_dates[1]

            if end_date < start_date:

                return (
                    "⚠️ Le date inserite non sono valide.\n\n"
                    "La data di fine/ritorno non può essere "
                    "precedente alla data di partenza.\n\n"
                    "Inserisci nuovamente le date corrette."
                )

        return None

    # ========================================================
    # CONTROLLO NUMERO VIAGGIATORI
    # ========================================================

    def validate_travelers(self, messages: list) -> str | None:
        """
        Verifica che siano stati specificati gli adulti
        e i bambini prima di iniziare una ricerca.

        Regole:

        - "Siamo 2 adulti" -> valido
        - "Siamo 2 adulti e 2 bambini" -> valido
        - "Siamo in 4" -> chiede la suddivisione
        - "Viaggiamo in 4 persone" -> chiede la suddivisione
        - nessuna informazione -> chiede adulti/bambini

        Questo metodo NON controlla ancora le età dei bambini.
        """

        user_messages = self.get_user_messages(messages)

        if not user_messages:
            return None

        adults_count = None
        children_count = None
        total_people = None

        number_words = {
            "uno": 1,
            "una": 1,
            "due": 2,
            "tre": 3,
            "quattro": 4,
            "cinque": 5,
            "sei": 6,
            "sette": 7,
            "otto": 8,
            "nove": 9,
            "dieci": 10,
        }

        for content in user_messages:

            text = content.lower()

            # =================================================
            # ADULTI NUMERICI
            # =================================================

            adult_match = re.search(
                r"\b(\d+)\s+(?:adulti|adulto)\b",
                text
            )

            if adult_match:

                adults_count = int(
                    adult_match.group(1)
                )

            # =================================================
            # ADULTI SCRITTI IN LETTERE
            # =================================================

            for word, number in number_words.items():

                adult_word_match = re.search(
                    rf"\b{word}\s+(?:adulti|adulto)\b",
                    text
                )

                if adult_word_match:

                    adults_count = number
                    break

            # =================================================
            # BAMBINI NUMERICI
            # =================================================

            children_match = re.search(
                r"\b(\d+)\s+(?:bambini|bambine|bimbi|bimbe)\b",
                text
            )

            if children_match:

                children_count = int(
                    children_match.group(1)
                )

            # =================================================
            # BAMBINI SCRITTI IN LETTERE
            # =================================================

            for word, number in number_words.items():

                child_word_match = re.search(
                    rf"\b{word}\s+(?:bambini|bambine|bimbi|bimbe)\b",
                    text
                )

                if child_word_match:

                    children_count = number
                    break

            # =================================================
            # NUMERO TOTALE
            # =================================================

            total_match = re.search(
                r"\b(?:siamo|viaggiamo|partiamo)"
                r"\s+(?:in\s+)?(\d+)"
                r"(?:\s+persone)?\b",
                text
            )

            if total_match:

                total_people = int(
                    total_match.group(1)
                )

        # ====================================================
        # ADULTI + BAMBINI
        # ====================================================

        if adults_count is not None and children_count is not None:

            return None

        # ====================================================
        # SOLO ADULTI
        #
        # "Siamo 2 adulti"
        #
        # Consideriamo automaticamente 0 bambini.
        # ====================================================

        if adults_count is not None and children_count is None:

            return None

        # ====================================================
        # SOLO BAMBINI
        # ====================================================

        if adults_count is None and children_count is not None:

            return (
                "👥 Perfetto! Mi dici anche quanti adulti "
                "viaggeranno?"
            )

        # ====================================================
        # SOLO NUMERO TOTALE
        # ====================================================

        if total_people is not None:

            return (
                "👥 Perfetto! Prima di organizzare il viaggio "
                "mi dici quanti sono gli adulti e quanti sono "
                "i bambini?"
            )

        # ====================================================
        # NESSUNA INFORMAZIONE
        # ====================================================

        return (
            "👥 Prima di organizzare il viaggio, mi dici "
            "quante persone viaggeranno? "
            "Mi servono il numero di adulti e di bambini."
        )

    # ========================================================
    # CONTROLLO BAMBINI
    # ========================================================

    def validate_children_ages(self, messages: list) -> str | None:

        user_messages = self.get_user_messages(messages)

        if not user_messages:
            return None

        number_words = {
            "uno": 1,
            "una": 1,
            "due": 2,
            "tre": 3,
            "quattro": 4,
            "cinque": 5,
            "sei": 6,
            "sette": 7,
            "otto": 8,
            "nove": 9,
            "dieci": 10,
        }

        children_count = None

        for content in user_messages:

            text = content.lower()

            numeric_match = re.search(
                r"\b(\d+)\s+(?:bambini|bambine|bimbi|bimbe)\b",
                text
            )

            if numeric_match:

                children_count = int(
                    numeric_match.group(1)
                )

                continue

            for word, number in number_words.items():

                word_match = re.search(
                    rf"\b{word}\s+(?:bambini|bambine|bimbi|bimbe)\b",
                    text
                )

                if word_match:

                    children_count = number
                    break

        if children_count is None or children_count <= 0:

            return None

        # ====================================================
        # CERCA ETÀ
        # ====================================================

        ages = []

        for content in user_messages:

            text = content.lower()

            age_matches = re.findall(
                r"\b(\d{1,2})\s*anni\b",
                text
            )

            for age_string in age_matches:

                age = int(age_string)

                if 0 <= age <= 17:

                    ages.append(age)

        # ====================================================
        # RISPOSTE BREVI
        #
        # "4 e 9"
        # "4 e 9 anni"
        # "4, 9"
        # ====================================================

        for content in user_messages:

            text = content.lower().strip()

            short_age_match = re.fullmatch(
                r"(\d{1,2})\s*(?:anni?)?\s*"
                r"(?:e|,)\s*"
                r"(\d{1,2})\s*(?:anni?)?",
                text
            )

            if short_age_match:

                first_age = int(
                    short_age_match.group(1)
                )

                second_age = int(
                    short_age_match.group(2)
                )

                if 0 <= first_age <= 17:

                    ages.append(first_age)

                if 0 <= second_age <= 17:

                    ages.append(second_age)

        # ====================================================
        # FORMATO:
        #
        # "uno ha 4 anni e l'altro 9 anni"
        # ====================================================

        for content in user_messages:

            text = content.lower()

            relative_match = re.search(
                r"(?:uno|una|il primo|la prima)"
                r".{0,40}?"
                r"(\d{1,2})\s*anni?"
                r".{0,60}?"
                r"(?:l'altro|l'altra|il secondo|la seconda)"
                r".{0,30}?"
                r"(\d{1,2})\s*anni?",
                text
            )

            if relative_match:

                first_age = int(
                    relative_match.group(1)
                )

                second_age = int(
                    relative_match.group(2)
                )

                if 0 <= first_age <= 17:

                    ages.append(first_age)

                if 0 <= second_age <= 17:

                    ages.append(second_age)

        # ====================================================
        # NON ELIMINIAMO I DUPLICATI
        # ====================================================

        valid_ages = [
            age
            for age in ages
            if 0 <= age <= 17
        ]

        # ====================================================
        # VERIFICA ETÀ MANCANTI
        # ====================================================

        if len(valid_ages) < children_count:

            missing = children_count - len(valid_ages)

            if children_count == 1:

                return (
                    "👶 Prima di procedere mi serve sapere "
                    "l'età del bambino."
                )

            if children_count == 2:

                if missing == 2:

                    return (
                        "👶 Prima di procedere mi serve sapere "
                        "l’età dei due bambini."
                    )

                return (
                    "👶 Mi dici anche l’età dell’altro bambino?"
                )

            return (
                "👶 Prima di procedere mi serve sapere "
                "l’età di tutti i bambini. "
                f"Mancano ancora {missing} età."
            )

        return None

    # ========================================================
    # IDENTIFICAZIONE TIPO DI RICHIESTA
    # ========================================================

    def detect_request_type(self, messages: list) -> str:

        user_messages = [
            content.lower()
            for content in self.get_user_messages(messages)
        ]

        if not user_messages:

            return "normal"

        complete_trip_keywords = [
            "viaggio completo",
            "organizza un viaggio",
            "organizzami un viaggio",
            "organizzare un viaggio",
            "pianifica un viaggio",
            "pianificare un viaggio",
            "programma un viaggio",
            "programmare un viaggio",
            "vacanza completa",
            "organizza la vacanza",
            "organizzare la vacanza",
            "pianifica la vacanza",
            "pianificare la vacanza",
        ]

        flight_keywords = [
            "volo",
            "voli",
            "aereo",
            "aerei",
            "volare",
            "volo da",
            "volo per",
            "biglietto aereo",
            "biglietti aerei",
        ]

        hotel_keywords = [
            "hotel",
            "albergo",
            "alloggio",
            "alloggi",
            "dove dormire",
            "dove soggiornare",
            "soggiorno",
            "camera",
            "camere",
        ]

        itinerary_keywords = [
            "itinerario",
            "itinerari",
            "cosa vedere",
            "cosa visitare",
            "cosa fare",
            "programma di viaggio",
            "programma del viaggio",
            "giorno per giorno",
            "piano di viaggio",
        ]

        historical_keywords = [
            "storia",
            "storico",
            "storica",
            "storici",
            "storiche",
            "storia di",
            "origine",
            "origini",
            "quando è stato costruito",
            "quando fu costruito",
            "chi lo ha costruito",
            "chi l'ha costruito",
            "cultura",
            "culturale",
            "culturali",
            "monumento",
            "monumenti",
            "significato storico",
        ]

        for content in reversed(user_messages):

            if any(
                keyword in content
                for keyword in complete_trip_keywords
            ):

                return "complete_trip"

            if any(
                keyword in content
                for keyword in flight_keywords
            ):

                return "flight"

            if any(
                keyword in content
                for keyword in hotel_keywords
            ):

                return "hotel"

            if any(
                keyword in content
                for keyword in itinerary_keywords
            ):

                return "itinerary"

            if any(
                keyword in content
                for keyword in historical_keywords
            ):

                return "historical"

        return "normal"

    # ========================================================
    # CREAZIONE AGENTE CON TOOL OBBLIGATORIO
    # ========================================================

    def create_forced_agent(self, tool):

        forced_model = self.model.bind_tools(
            [tool],
            tool_choice="required"
        )

        return create_react_agent(
            forced_model,
            [tool]
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def build_system_prompt(self, current_datetime):

        SYSTEM_PROMPT = f"""
Sei un travel planner. Il tuo compito è organizzare
il viaggio per l'utente.

Aggiungi delle emojis per rendere il tuo output più interessante.

La data e l'ora attuale sono:
{current_datetime}


============================================================
REGOLA FONDAMENTALE: UTILIZZO DEI TOOL
============================================================

I tool disponibili sono:

- flights_finder
- hotels_finder
- chain_travel_plan
- chain_historical_expert

Quando una richiesta dell'utente richiede informazioni
che devono essere ottenute tramite uno dei tool disponibili,
DEVI utilizzare il relativo tool PRIMA di fornire la risposta
all'utente.

Il tool deve essere effettivamente eseguito.

NON devi rispondere direttamente utilizzando informazioni
inventate, ipotizzate o basate solamente sulla conoscenza
interna del modello quando il dato richiesto deve essere
ottenuto tramite un tool.


============================================================
REGOLE OBBLIGATORIE PER I VIAGGIATORI
============================================================

Prima di iniziare QUALSIASI ricerca relativa a:

- voli;
- hotel;
- itinerari;
- viaggi completi;

devi sapere quante persone viaggeranno.

Devi conoscere:

- numero di adulti;
- numero di bambini.

Se l'utente indica solamente il numero totale di persone,
NON devi assumere che siano tutti adulti.

Devi chiedere quanti sono adulti e quanti sono bambini.

Se l'utente dice per esempio:

"Siamo 4 persone"

devi chiedere:

"👥 Quante sono le persone adulte e quanti sono i bambini?"

Se l'utente dice:

"Siamo 2 adulti"

puoi considerare automaticamente:

- 2 adulti;
- 0 bambini.

Se sono presenti bambini devi conoscere l'età di OGNI bambino.

Esempio:

"Siamo 2 adulti e 2 bambini"

deve comportare una richiesta delle età dei due bambini.

NON devi iniziare alcuna ricerca finché non sono
disponibili tutte le età dei bambini.


============================================================
REGOLE OBBLIGATORIE PER LE DATE
============================================================

Le date del viaggio sono informazioni OBBLIGATORIE.

NON devi mai inventare una data.

NON devi mai assumere una data sulla base della data attuale.

NON devi mai iniziare una ricerca di voli senza conoscere
la data di partenza.

Per una richiesta di volo devi conoscere almeno:

- aeroporto di partenza;
- destinazione;
- data di partenza;
- numero di adulti;
- numero di bambini;
- età dei bambini, se presenti.

Se manca la data di partenza devi fermarti e chiedere:

"📅 Per poter cercare i voli mi serve sapere la data
di partenza."

La data deve essere fornita nel formato:

YYYY-MM-DD

Esempio:

2026-11-15

Se l'utente fornisce una sola data per un viaggio completo,
devi chiedere anche la data di ritorno/fine viaggio.

Per un viaggio completo devi conoscere:

- data di partenza;
- data di ritorno/fine viaggio.

La data di ritorno non può essere precedente alla data
di partenza.

NON chiamare alcun tool finché le date obbligatorie
non sono disponibili.


============================================================
REGOLE OBBLIGATORIE PER I TOOL
============================================================

- Usa `flights_finder` per cercare i voli.
- Usa `hotels_finder` per cercare gli hotel.
- Usa `chain_travel_plan` per creare l'itinerario.
- Usa `chain_historical_expert` quando sono richieste
  informazioni storiche o culturali.

Se una richiesta richiede più informazioni che devono essere
ottenute tramite tool diversi, utilizza tutti i tool necessari.

Per un viaggio completo:

1. `flights_finder`
2. `hotels_finder`
3. `chain_travel_plan`

NON saltare un tool.

NON fornire risultati di voli senza aver utilizzato
`flights_finder`.

NON fornire risultati di hotel senza aver utilizzato
`hotels_finder`.

NON creare un itinerario senza aver utilizzato
`chain_travel_plan`.


============================================================
GESTIONE DEGLI ERRORI DEI TOOL
============================================================

Se un tool viene eseguito ma restituisce un errore:

- informa l'utente del problema;
- non inventare un risultato alternativo;
- non sostituire il risultato mancante con informazioni
  inventate dal modello.


============================================================
REGOLA FONDAMENTALE: AEROPORTO DI PARTENZA
============================================================

NON DEVI MAI INVENTARE, INDOVINARE O ASSUMERE
L'AEROPORTO DI PARTENZA.

L'aeroporto di partenza deve essere fornito
ESPRESSAMENTE dall'utente.

Prima di chiamare `flights_finder`, devi verificare
che l'utente abbia indicato esplicitamente l'aeroporto
dal quale vuole partire.

Se l'utente dice:

"Parto da Milano"

NON devi scegliere automaticamente MXP o LIN.

Devi chiedere quale aeroporto vuole utilizzare.

Se l'utente dice:

"Voglio andare a Roma"

ma non indica l'aeroporto di partenza:

NON devi chiamare `flights_finder`.

Devi chiedere:

"✈️ Per poter cercare i voli mi serve sapere
da quale aeroporto vuoi partire."

NON utilizzare la posizione geografica dell'utente
per scegliere automaticamente l'aeroporto.


============================================================
AEROPORTO DI ARRIVO
============================================================

Anche per l'aeroporto di arrivo non devi inventare
informazioni.

Se la destinazione non permette di determinare chiaramente
l'aeroporto appropriato, chiedi chiarimenti.


============================================================
INFORMAZIONI MANCANTI
============================================================

Quando manca un'informazione obbligatoria:

1. NON chiamare il tool.
2. NON inventare il valore.
3. Chiedi direttamente all'utente l'informazione mancante.

Devi verificare le informazioni necessarie PRIMA
dell'esecuzione del tool.


============================================================
COMPORTAMENTO PER UN VIAGGIO COMPLETO
============================================================

Prima di eseguire qualsiasi tool devi avere:

- numero di adulti;
- numero di bambini;
- età di tutti i bambini, se presenti;
- aeroporto di partenza;
- destinazione;
- data di partenza;
- data di ritorno/fine viaggio.

Se manca una di queste informazioni, chiedila all'utente
e NON eseguire ancora i tool.

Quando tutte le informazioni sono disponibili:

1. utilizza `flights_finder`;
2. utilizza `hotels_finder`;
3. utilizza `chain_travel_plan`.

La risposta finale deve essere costruita esclusivamente
sui risultati realmente ottenuti dai tool.


============================================================
ESEMPI DI OUTPUT
============================================================

ATTENZIONE:

Gli esempi servono esclusivamente a mostrare il formato.

I dati presenti negli esempi NON sono dati reali.

NON utilizzare mai nomi, compagnie aeree, prezzi, date,
hotel o disponibilità presenti negli esempi come risultati reali.

I dati reali devono provenire dai tool.


============================================================
ESEMPIO OUTPUT VOLI
============================================================

{FLIGHTS_OUTPUT}


============================================================
ESEMPIO OUTPUT HOTEL
============================================================

{HOTELS_OUTPUT}


============================================================
ESEMPIO OUTPUT VIAGGIO
============================================================

{TRAVEL_PLAN_OUTPUT}
"""

        return SYSTEM_PROMPT

    # ========================================================
    # ESECUZIONE TOOL OBBLIGATORIO
    # ========================================================

    def run_forced_tool(
        self,
        messages: list,
        system_prompt: str,
        tool
    ):

        agent = self.create_forced_agent(tool)

        conversation_history = [
            {
                "role": "system",
                "content": system_prompt
            }
        ] + messages

        response = agent.invoke(
            {
                "messages": conversation_history
            }
        )

        return response["messages"][1:]

    # ========================================================
    # RUN
    # ========================================================

    def run(self, messages: list):

        # ====================================================
        # DATA E ORA ATTUALE
        # ====================================================

        current_datetime = datetime.now()

        # ====================================================
        # IDENTIFICAZIONE DEL TIPO DI RICHIESTA
        #
        # Deve avvenire PRIMA delle validazioni.
        # ====================================================

        request_type = self.detect_request_type(
            messages
        )

        # ====================================================
        # CONVERSAZIONE NORMALE
        # ====================================================

        if request_type == "normal":

            SYSTEM_PROMPT = self.build_system_prompt(
                current_datetime
            )

            conversation_history = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ] + messages

            normal_agent = create_react_agent(
                self.model,
                self.tools
            )

            response = normal_agent.invoke(
                {
                    "messages": conversation_history
                }
            )

            return response["messages"][1:]

        # ====================================================
        # CONTROLLO DATE
        #
        # ORA VIENE FATTO DOPO AVER CAPITO CHE SI TRATTA
        # DI UNA RICHIESTA DI VIAGGIO.
        # ====================================================

        date_error = self.validate_dates(
            messages,
            request_type
        )

        if date_error:

            return [
                {
                    "role": "assistant",
                    "content": date_error
                }
            ]

        # ====================================================
        # CONTROLLO NUMERO VIAGGIATORI
        #
        # NON NECESSARIO PER UNA SEMPLICE DOMANDA STORICA.
        # ====================================================

        travel_request_types = [
            "flight",
            "hotel",
            "itinerary",
            "complete_trip",
        ]

        if request_type in travel_request_types:

            travelers_error = self.validate_travelers(
                messages
            )

            if travelers_error:

                return [
                    {
                        "role": "assistant",
                        "content": travelers_error
                    }
                ]

            # =================================================
            # CONTROLLO ETÀ BAMBINI
            # =================================================

            children_error = self.validate_children_ages(
                messages
            )

            if children_error:

                return [
                    {
                        "role": "assistant",
                        "content": children_error
                    }
                ]

        # ====================================================
        # SYSTEM PROMPT
        # ====================================================

        SYSTEM_PROMPT = self.build_system_prompt(
            current_datetime
        )

        # ====================================================
        # VOLO
        # ====================================================

        if request_type == "flight":

            return self.run_forced_tool(
                messages,
                SYSTEM_PROMPT,
                flights_finder
            )

        # ====================================================
        # HOTEL
        # ====================================================

        if request_type == "hotel":

            return self.run_forced_tool(
                messages,
                SYSTEM_PROMPT,
                hotels_finder
            )

        # ====================================================
        # ITINERARIO
        # ====================================================

        if request_type == "itinerary":

            return self.run_forced_tool(
                messages,
                SYSTEM_PROMPT,
                chain_travel_plan
            )

        # ====================================================
        # STORICO / CULTURA
        # ====================================================

        if request_type == "historical":

            return self.run_forced_tool(
                messages,
                SYSTEM_PROMPT,
                chain_historical_expert
            )

        # ====================================================
        # VIAGGIO COMPLETO
        # ====================================================

        if request_type == "complete_trip":

            # =================================================
            # 1. FLIGHTS
            # =================================================

            flights_result = self.run_forced_tool(
                messages,
                SYSTEM_PROMPT,
                flights_finder
            )

            flight_text = "\n".join(
                str(message.get("content", ""))
                for message in flights_result
                if isinstance(message, dict)
            )

            # =================================================
            # 2. HOTELS
            # =================================================

            hotel_messages = messages + [
                {
                    "role": "assistant",
                    "content": (
                        "Risultato ottenuto da flights_finder:\n\n"
                        + flight_text
                    )
                }
            ]

            hotels_result = self.run_forced_tool(
                hotel_messages,
                SYSTEM_PROMPT,
                hotels_finder
            )

            hotel_text = "\n".join(
                str(message.get("content", ""))
                for message in hotels_result
                if isinstance(message, dict)
            )

            # =================================================
            # 3. ITINERARIO
            # =================================================

            itinerary_messages = hotel_messages + [
                {
                    "role": "assistant",
                    "content": (
                        "Risultato ottenuto da hotels_finder:\n\n"
                        + hotel_text
                    )
                }
            ]

            itinerary_result = self.run_forced_tool(
                itinerary_messages,
                SYSTEM_PROMPT,
                chain_travel_plan
            )

            itinerary_text = "\n".join(
                str(message.get("content", ""))
                for message in itinerary_result
                if isinstance(message, dict)
            )

            # =================================================
            # 4. RISPOSTA FINALE
            # =================================================

            final_prompt = """
Costruisci ora la risposta finale per l'utente utilizzando
ESCLUSIVAMENTE i risultati realmente ottenuti dai tool.

Sono stati eseguiti obbligatoriamente:

1. flights_finder
2. hotels_finder
3. chain_travel_plan

Non inventare informazioni.

Non modificare i risultati dei tool.

Presenta la risposta in modo chiaro, ordinato e leggibile,
utilizzando markdown ed emojis come richiesto dal system prompt.

Includi:

- voli;
- hotel;
- itinerario.

Se uno dei tool ha restituito un errore, comunicalo
chiaramente invece di inventare un risultato.
"""

            final_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        "Richiesta originale dell'utente:\n\n"
                        + "\n".join(
                            self.get_user_messages(messages)
                        )
                        + "\n\n"
                        + final_prompt
                        + "\n\n"
                        + "RISULTATO FLIGHTS_FINDER:\n"
                        + flight_text
                        + "\n\n"
                        + "RISULTATO HOTELS_FINDER:\n"
                        + hotel_text
                        + "\n\n"
                        + "RISULTATO CHAIN_TRAVEL_PLAN:\n"
                        + itinerary_text
                    )
                }
            ]

            final_response = self.model.invoke(
                final_messages
            )

            return [
                {
                    "role": "assistant",
                    "content": final_response.content
                }
            ]

        # ====================================================
        # FALLBACK
        # ====================================================

        conversation_history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ] + messages

        fallback_agent = create_react_agent(
            self.model,
            self.tools
        )

        response = fallback_agent.invoke(
            {
                "messages": conversation_history
            }
        )

        return response["messages"][1:]