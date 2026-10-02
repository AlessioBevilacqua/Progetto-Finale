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
    - Ora di ritorno: 14:30
    - Durata del volo: 1h 30m

    Inserisci il link di Google per la prenotazione se possibile.
    
    #### Altri opzioni disponibili:

    - Compagnia aerea: Ryanair
    - Data di partenza: 2024-12-13
    - Ora di partenza: 10:00
    - Durata del volo: 1h 30m

    - Compagnia aerea: Ryanair
    - Data di ritorno: 2024-12-19
    - Ora di ritorno: 14:30
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

        self.agent_executor = create_react_agent(
            self.model,
            self.tools
        )

    # ========================================================
    # CONTROLLO DATE
    # ========================================================

    def validate_dates(self, messages: list) -> str | None:
        """
        Controlla se nei messaggi dell'utente sono presenti
        date nel formato YYYY-MM-DD e verifica che non siano
        nel passato.
        """

        today = date.today()

        # Cerchiamo date nel formato YYYY-MM-DD
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"

        for message in messages:

            # Consideriamo solo messaggi testuali
            if not isinstance(message, dict):
                continue

            if message.get("role") != "user":
                continue

            content = message.get("content", "")

            if not isinstance(content, str):
                continue

            found_dates = re.findall(
                date_pattern,
                content
            )

            for date_string in found_dates:

                try:
                    requested_date = datetime.strptime(
                        date_string,
                        "%Y-%m-%d"
                    ).date()

                except ValueError:
                    continue

                if requested_date < today:
                    return (
                        "⚠️ Le date inserite non sono valide.\n\n"
                        f"La data **{date_string}** è già passata. "
                        f"Oggi è **{today}**.\n\n"
                        "Inserisci una data futura per poter "
                        "organizzare il viaggio."
                    )

        return None

    # ========================================================
    # RUN
    # ========================================================

    def run(self, messages: list):

        # Data attuale aggiornata ad ogni richiesta
        current_datetime = datetime.now()

        # ----------------------------------------------------
        # CONTROLLO DATE PRIMA DELL'AGENTE
        # ----------------------------------------------------

        date_error = self.validate_dates(messages)

        if date_error:
            return [
                {
                    "role": "assistant",
                    "content": date_error
                }
            ]

        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        SYSTEM_PROMPT = f"""
Sei un travel planner. Il tuo compito è organizzare
il viaggio per l'utente.

Aggiungi delle emojis per rendere il tuo output più interessante.

La data e l'ora attuale sono:
{current_datetime}

IMPORTANTE:

- Non inventare voli, hotel o disponibilità.
- Usa flights_finder per cercare i voli.
- Usa hotels_finder per cercare gli hotel.
- Usa chain_travel_plan per creare l'itinerario.
- Se un tool restituisce un errore relativo alle date,
  informa l'utente del problema.
- Non sostituire una ricerca reale fallita con dati inventati.
- Non creare un itinerario alternativo se le date richieste
  dall'utente non sono valide.

Esempio Output Voli:

{FLIGHTS_OUTPUT}

Esempio Output Hotel:

{HOTELS_OUTPUT}

Esempio Output Viaggio:

{TRAVEL_PLAN_OUTPUT}
"""

        # ----------------------------------------------------
        # CONVERSATION HISTORY
        # ----------------------------------------------------

        conversation_history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ] + messages

        # ----------------------------------------------------
        # ESECUZIONE AGENTE
        # ----------------------------------------------------

        response = self.agent_executor.invoke(
            {
                "messages": conversation_history
            }
        )

        return response["messages"][1:]