import os
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch


load_dotenv()


# ============================================================
# INPUT DEL TOOL
# ============================================================

class FlightsInput(BaseModel):

    departure_airport: str = Field(
        description=(
            "The IATA code of the departure airport. "
            "IMPORTANT: This airport MUST be explicitly provided "
            "by the user. NEVER guess, infer, assume, or invent "
            "the departure airport from the user's location, city, "
            "nationality, or any other context. "
            "If the user has not explicitly specified the departure "
            "airport, DO NOT call this tool. Ask the user which "
            "airport they want to depart from. "
            "Examples: MXP, LIN, FCO, BRI."
        )
    )

    arrival_airport: str = Field(
        description=(
            "The IATA code of the arrival airport. "
            "Use the airport explicitly specified by the user. "
            "Do not invent an airport. "
            "Examples: FCO, MXP, BRI."
        )
    )

    outbound_date: date = Field(
        description=(
            "The outbound flight date in YYYY-MM-DD format."
        )
    )

    return_date: date = Field(
        description=(
            "The return flight date in YYYY-MM-DD format."
        )
    )

    adults: Optional[int] = Field(
        default=1,
        ge=1,
        description="The number of adults."
    )

    children: Optional[int] = Field(
        default=0,
        ge=0,
        description="The number of children."
    )


# ============================================================
# TOOL
# ============================================================

@tool(args_schema=FlightsInput)
def flights_finder(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: date,
    return_date: date,
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
):
    """
    Searches for flights using Google Flights through SerpAPI.

    IMPORTANT:
    The departure airport must always be explicitly provided
    by the user. The tool must never guess or infer it.
    """

    print("=" * 80)
    print("flights_finder")
    print("=" * 80)

    # ========================================================
    # NORMALIZZAZIONE VALORI
    # ========================================================

    if adults is None:
        adults = 1

    if children is None:
        children = 0

    # ========================================================
    # NORMALIZZAZIONE AEROPORTI
    # ========================================================

    if departure_airport is None:
        departure_airport = ""

    if arrival_airport is None:
        arrival_airport = ""

    departure_airport = departure_airport.strip().upper()
    arrival_airport = arrival_airport.strip().upper()

    # ========================================================
    # VALIDAZIONE AEROPORTO DI PARTENZA
    # ========================================================

    if not departure_airport:
        print(
            "ERRORE: aeroporto di partenza non specificato."
        )

        return {
            "success": False,
            "error": (
                "⚠️ Per poter cercare i voli è necessario "
                "specificare l'aeroporto di partenza."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # VALIDAZIONE AEROPORTO DI ARRIVO
    # ========================================================

    if not arrival_airport:
        print(
            "ERRORE: aeroporto di arrivo non specificato."
        )

        return {
            "success": False,
            "error": (
                "⚠️ Per poter cercare i voli è necessario "
                "specificare l'aeroporto di arrivo."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # VALIDAZIONE FORMATO AEROPORTI
    # ========================================================

    if len(departure_airport) != 3:
        return {
            "success": False,
            "error": (
                "⚠️ L'aeroporto di partenza deve essere "
                "specificato tramite un codice IATA valido "
                "di 3 lettere, ad esempio MXP, LIN o FCO."
            ),
            "flights": [],
            "other_flights": [],
        }

    if len(arrival_airport) != 3:
        return {
            "success": False,
            "error": (
                "⚠️ L'aeroporto di arrivo deve essere "
                "specificato tramite un codice IATA valido "
                "di 3 lettere, ad esempio FCO, MXP o BRI."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # VALIDAZIONE DATE
    # ========================================================

    today = date.today()

    # --------------------------------------------------------
    # DATA DI PARTENZA NEL PASSATO
    # --------------------------------------------------------

    if outbound_date < today:

        return {
            "success": False,
            "error": (
                "⚠️ La data di partenza del volo non può essere "
                "nel passato. "
                f"La data indicata è {outbound_date}. "
                f"Inserisci una data a partire da {today}."
            ),
            "flights": [],
            "other_flights": [],
        }

    # --------------------------------------------------------
    # DATA DI RITORNO NEL PASSATO
    # --------------------------------------------------------

    if return_date < today:

        return {
            "success": False,
            "error": (
                "⚠️ La data di ritorno del volo non può essere "
                "nel passato. "
                f"La data indicata è {return_date}. "
                f"Inserisci una data a partire da {today}."
            ),
            "flights": [],
            "other_flights": [],
        }

    # --------------------------------------------------------
    # RITORNO PRIMA DELLA PARTENZA
    # --------------------------------------------------------

    if return_date < outbound_date:

        return {
            "success": False,
            "error": (
                "⚠️ Le date del volo non sono valide. "
                "La data di ritorno non può essere precedente "
                "alla data di partenza."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # API KEY SERPAPI
    # ========================================================

    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:

        print(
            "ERRORE: SERPAPI_API_KEY non configurata."
        )

        return {
            "success": False,
            "error": (
                "⚠️ La chiave SERPAPI_API_KEY non è configurata. "
                "Controlla il file .env."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # PARAMETRI SERPAPI
    # ========================================================

    query_params = {
        "api_key": api_key,
        "engine": "google_flights",
        "hl": "it",
        "gl": "it",
        "currency": "EUR",

        # IMPORTANTE:
        # Questi valori arrivano dal tool e NON vengono
        # determinati automaticamente.
        "departure_id": departure_airport,
        "arrival_id": arrival_airport,

        "outbound_date": outbound_date.isoformat(),
        "return_date": return_date.isoformat(),

        "adults": adults,
        "children": children,
    }

    # ========================================================
    # DEBUG
    # ========================================================

    print("-" * 80)
    print("FLIGHT SEARCH PARAMETERS")
    print(f"Departure airport: {departure_airport}")
    print(f"Arrival airport:   {arrival_airport}")
    print(f"Outbound date:     {outbound_date}")
    print(f"Return date:       {return_date}")
    print(f"Adults:            {adults}")
    print(f"Children:          {children}")
    print("-" * 80)

    # ========================================================
    # CHIAMATA SERPAPI
    # ========================================================

    try:

        search = GoogleSearch(query_params)

        response = search.get_dict()

        print(
            "SerpAPI response keys:",
            response.keys()
        )

        # ====================================================
        # ERRORE RESTITUITO DA SERPAPI
        # ====================================================

        if "error" in response:

            error_message = response["error"]

            print(
                "SERPAPI FLIGHTS ERROR:",
                error_message
            )

            return {
                "success": False,
                "error": (
                    "⚠️ Google Flights non ha trovato "
                    "risultati per la tratta e le date indicate."
                ),
                "details": error_message,
                "flights": [],
                "other_flights": [],
            }

        # ====================================================
        # ESTRAZIONE RISULTATI
        # ====================================================

        best_flights = response.get(
            "best_flights",
            []
        )

        other_flights = response.get(
            "other_flights",
            []
        )

        # ====================================================
        # NESSUN RISULTATO
        # ====================================================

        if not best_flights and not other_flights:

            print(
                "Google Flights non ha restituito "
                "alcun volo."
            )

            return {
                "success": False,
                "error": (
                    "⚠️ Google Flights non ha trovato voli "
                    "per la tratta e le date indicate."
                ),
                "flights": [],
                "other_flights": [],
            }

        # ====================================================
        # RISULTATI TROVATI
        # ====================================================

        print(
            f"Trovati {len(best_flights)} voli consigliati "
            f"e {len(other_flights)} altri voli."
        )

        print("=" * 80)

        return {
            "success": True,
            "flights": best_flights,
            "other_flights": other_flights,
        }

    # ========================================================
    # ERRORE GENERICO
    # ========================================================

    except Exception as e:

        print("=" * 80)
        print("FLIGHTS FINDER ERROR")
        print("=" * 80)

        print(repr(e))

        print("=" * 80)

        return {
            "success": False,
            "error": (
                "⚠️ Si è verificato un errore durante "
                "la ricerca dei voli."
            ),
            "details": str(e),
            "flights": [],
            "other_flights": [],
        }