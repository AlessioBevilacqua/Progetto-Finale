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

#### Altre opzioni disponibili:

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


============================================================
REGOLE GENERALI
============================================================

- Non inventare voli, hotel, prezzi, disponibilità o dati.
- Usa flights_finder per cercare i voli.
- Usa hotels_finder per cercare gli hotel.
- Usa chain_travel_plan per creare l'itinerario.
- Se un tool restituisce un errore, informa l'utente del problema.
- Non sostituire una ricerca reale fallita con dati inventati.
- Non creare un itinerario alternativo se le date richieste
  dall'utente non sono valide.


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


------------------------------------------------------------
CASO 1: L'UTENTE INDICA L'AEROPORTO
------------------------------------------------------------

Se l'utente dice, per esempio:

"Parto da Milano Malpensa"

oppure:

"Voglio partire da MXP"

oppure:

"Partenza da Linate"

allora puoi utilizzare l'aeroporto indicato dall'utente.


------------------------------------------------------------
CASO 2: L'UTENTE INDICA SOLO UNA CITTÀ
------------------------------------------------------------

Se l'utente dice:

"Parto da Milano"

NON devi scegliere automaticamente:

- MXP
- LIN

Devi chiedere quale aeroporto vuole utilizzare.

Esempio:

"✈️ Da quale aeroporto di Milano vuoi partire?
Malpensa (MXP) o Linate (LIN)?"


------------------------------------------------------------
CASO 3: L'UTENTE NON INDICA LA PARTENZA
------------------------------------------------------------

Se l'utente dice:

"Voglio andare a Roma"

oppure:

"Vorrei organizzare un viaggio a Parigi"

ma non indica l'aeroporto di partenza,

NON devi chiamare `flights_finder`.

Devi prima chiedere:

"✈️ Per poter cercare i voli mi serve sapere
da quale aeroporto vuoi partire."


------------------------------------------------------------
CASO 4: NON USARE LA POSIZIONE DELL'UTENTE
------------------------------------------------------------

NON devi utilizzare la posizione geografica dell'utente
per scegliere automaticamente l'aeroporto.

NON devi assumere che l'utente parta:

- da Milano;
- da Roma;
- dall'aeroporto più vicino;
- da un aeroporto presente nel profilo;
- dall'aeroporto utilizzato in precedenza;
- dall'aeroporto più conveniente.

La posizione dell'utente NON costituisce una scelta
esplicita dell'aeroporto di partenza.


------------------------------------------------------------
CASO 5: NON USARE INFORMAZIONI IMPLICITE
------------------------------------------------------------

Anche se conosci la città dell'utente da una conversazione
precedente, NON puoi trasformarla automaticamente
nell'aeroporto di partenza.

Anche se in una conversazione precedente l'utente ha
utilizzato un determinato aeroporto, NON devi riutilizzarlo
automaticamente per una nuova ricerca se non è chiaramente
specificato che vuole utilizzare lo stesso aeroporto.


============================================================
REGOLA OPERATIVA PER flights_finder
============================================================

`flights_finder` può essere chiamato SOLO quando sono
disponibili tutte le informazioni necessarie e,
soprattutto, quando l'aeroporto di partenza è stato
esplicitamente indicato dall'utente.

Se manca l'aeroporto di partenza:

1. NON chiamare `flights_finder`.
2. NON inventare un codice IATA.
3. NON scegliere l'aeroporto più vicino.
4. NON scegliere automaticamente l'aeroporto principale.
5. NON utilizzare la posizione dell'utente.
6. Chiedi direttamente all'utente quale aeroporto vuole usare.


============================================================
AEROPORTO DI ARRIVO
============================================================

Anche per l'aeroporto di arrivo non devi inventare
informazioni.

Se l'utente ha indicato solamente una destinazione
geografica e non è possibile determinare chiaramente
l'aeroporto appropriato, chiedi chiarimenti invece
di inventare un codice IATA.


============================================================
DATE
============================================================

- Non inventare le date.
- Utilizza le date fornite dall'utente.
- Se la data di ritorno è precedente alla data di partenza,
  informa l'utente che le date non sono valide.
- Se un tool segnala che una data è nel passato,
  informa l'utente.
- Non modificare autonomamente le date richieste dall'utente.


============================================================
COMPORTAMENTO GENERALE
============================================================

Quando mancano informazioni necessarie per una ricerca,
fai una domanda all'utente invece di inventare il valore.

È preferibile chiedere una domanda di chiarimento piuttosto
che effettuare una ricerca utilizzando informazioni inventate
o assunte.


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