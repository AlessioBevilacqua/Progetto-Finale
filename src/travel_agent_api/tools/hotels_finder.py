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

class HotelsInput(BaseModel):

    q: str = Field(
        description="Location of the hotel."
    )

    check_in_date: date = Field(
        description="The check-in date (YYYY-MM-DD)."
    )

    check_out_date: date = Field(
        description="The check-out date (YYYY-MM-DD)."
    )

    adults: Optional[int] = Field(
        default=1,
        ge=1,
        description="Number of adults."
    )

    children: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of children."
    )

    children_ages: Optional[list[int]] = Field(
        default=None,
        description=(
            "Ages of all children. "
            "The number of ages must match the number of children. "
            "For example, if there are 2 children aged 8 and 10, "
            "use [8, 10]."
        )
    )

    hotel_class: Optional[int] = Field(
        default=2,
        ge=2,
        le=5,
        description="Hotel class from 2 to 5."
    )


# ============================================================
# TOOL
# ============================================================

@tool(args_schema=HotelsInput)
def hotels_finder(
    q: str,
    check_in_date: date,
    check_out_date: date,
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
    children_ages: Optional[list[int]] = None,
    hotel_class: Optional[int] = 2,
):
    """
    Searches for hotels using Google Hotels through SerpAPI.
    """

    # ========================================================
    # DATA ODIERNA
    # ========================================================

    today = date.today()

    # ========================================================
    # CONTROLLO DATE
    # ========================================================

    if check_in_date < today:
        return (
            "⚠️ La data di check-in non è valida. "
            f"La data {check_in_date} è già passata. "
            f"Inserisci una data a partire da {today}."
        )

    if check_out_date < today:
        return (
            "⚠️ La data di check-out non è valida. "
            f"La data {check_out_date} è già passata. "
            f"Inserisci una data a partire da {today}."
        )

    if check_out_date <= check_in_date:
        return (
            "⚠️ Le date dell'hotel non sono valide. "
            "La data di check-out deve essere successiva "
            "alla data di check-in."
        )

    # ========================================================
    # CONTROLLO NUMERO BAMBINI
    # ========================================================

    if children is None:
        children = 0

    if children > 0:

        if not children_ages:
            return (
                "⚠️ Per cercare un hotel con bambini è necessario "
                "conoscere l'età di ogni bambino."
            )

        if len(children_ages) != children:
            return (
                f"⚠️ Sono stati indicati {children} bambini, "
                f"ma sono state specificate {len(children_ages)} età. "
                "È necessario specificare l'età di ogni bambino."
            )

        # Controllo che le età siano valide
        for age in children_ages:

            if age < 0 or age > 17:
                return (
                    f"⚠️ L'età del bambino ({age}) non è valida. "
                    "L'età deve essere compresa tra 0 e 17 anni."
                )

    else:
        # Se non ci sono bambini non mandiamo children_ages
        children_ages = None

    # ========================================================
    # API KEY
    # ========================================================

    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        return (
            "⚠️ SERPAPI_API_KEY non configurata. "
            "Controlla il file .env."
        )

    # ========================================================
    # PARAMETRI SERPAPI
    # ========================================================

    query_params = {
        "api_key": api_key,
        "engine": "google_hotels",
        "q": q,
        "check_in_date": check_in_date.isoformat(),
        "check_out_date": check_out_date.isoformat(),
        "adults": adults,
        "children": children,
        "hotel_class": hotel_class,
        "currency": "EUR",
        "hl": "it",
        "gl": "it",
        "num": 5,
    }

    # ========================================================
    # ETÀ DEI BAMBINI
    # ========================================================

    if children > 0 and children_ages:

        query_params["children_ages"] = ",".join(
            str(age)
            for age in children_ages
        )

    # ========================================================
    # RICERCA SERPAPI
    # ========================================================

    try:

        search = GoogleSearch(query_params)

        results = search.get_dict()

        # ====================================================
        # LOG
        # ====================================================

        print("=" * 80)
        print("hotels_finder")
        print("=" * 80)

        print(
            "SerpAPI response keys:",
            results.keys()
        )

        # ====================================================
        # ERRORE SERPAPI
        # ====================================================

        if "error" in results:

            error_message = results["error"]

            print(
                "SERPAPI HOTELS ERROR:",
                error_message
            )

            return (
                "⚠️ Non è stato possibile trovare hotel "
                "per le date e i parametri indicati. "
                "Prova a modificare le date, la destinazione "
                "o i criteri di ricerca."
            )

        # ====================================================
        # PROPRIETÀ HOTEL
        # ====================================================

        properties = results.get(
            "properties",
            []
        )

        if not properties:

            return (
                "⚠️ Non sono stati trovati hotel disponibili "
                "per le date e i parametri indicati."
            )

        # ====================================================
        # RISULTATI
        # ====================================================

        hotels = []

        for hotel in properties[:5]:

            hotels.append(hotel)

        print(
            f"Trovati {len(hotels)} hotel."
        )

        print("=" * 80)

        return hotels

    # ========================================================
    # GESTIONE ERRORI
    # ========================================================

    except Exception as e:

        print("=" * 80)
        print("HOTELS FINDER ERROR")
        print("=" * 80)
        print(repr(e))
        print("=" * 80)

        return (
            "⚠️ Si è verificato un errore durante "
            "la ricerca degli hotel."
        )