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
            "The departure airport IATA code. "
            "Example: MXP, FCO, BRI."
        )
    )

    arrival_airport: str = Field(
        description=(
            "The arrival airport IATA code. "
            "Example: FCO, MXP, BRI."
        )
    )

    outbound_date: date = Field(
        description="The outbound flight date (YYYY-MM-DD)."
    )

    return_date: date = Field(
        description="The return flight date (YYYY-MM-DD)."
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
    """

    print("=" * 80)
    print("flights_finder")
    print("=" * 80)

    # ========================================================
    # NORMALIZZAZIONE
    # ========================================================

    if adults is None:
        adults = 1

    if children is None:
        children = 0

    departure_airport = departure_airport.strip().upper()
    arrival_airport = arrival_airport.strip().upper()

    # ========================================================
    # VALIDAZIONE AEROPORTI
    # ========================================================

    if not departure_airport:
        return {
            "success": False,
            "error": (
                "⚠️ Non è stato specificato l'aeroporto "
                "di partenza."
            ),
            "flights": [],
            "other_flights": [],
        }

    if not arrival_airport:
        return {
            "success": False,
            "error": (
                "⚠️ Non è stato specificato l'aeroporto "
                "di arrivo."
            ),
            "flights": [],
            "other_flights": [],
        }

    # ========================================================
    # VALIDAZIONE DATE
    # ========================================================

    today = date.today()

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
    # API KEY
    # ========================================================

    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        print("SERPAPI_API_KEY non configurata.")

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

        print("SerpAPI response keys:", response.keys())

        # ====================================================
        # ERRORE SERPAPI
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
                    "⚠️ Google Flights non ha trovato risultati "
                    "per la tratta e le date indicate."
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