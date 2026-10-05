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
        description="Location where the user wants to find hotels."
    )

    check_in_date: date = Field(
        description="Hotel check-in date (YYYY-MM-DD)."
    )

    check_out_date: date = Field(
        description="Hotel check-out date (YYYY-MM-DD)."
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
            "For example, 2 children aged 8 and 10 means [8, 10]."
        )
    )

    hotel_class: Optional[int] = Field(
        default=2,
        ge=2,
        le=5,
        description="Minimum hotel class from 2 to 5 stars."
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

    The tool intentionally returns only a small, filtered list
    of useful hotel information instead of the complete SerpAPI
    response, so the AI agent can process the results reliably.
    """

    # ========================================================
    # NORMALIZZAZIONE PARAMETRI
    # ========================================================

    if adults is None:
        adults = 1

    if children is None:
        children = 0

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
            f"La data {check_in_date.isoformat()} è già passata. "
            f"Inserisci una data a partire da {today.isoformat()}."
        )

    if check_out_date < today:
        return (
            "⚠️ La data di check-out non è valida. "
            f"La data {check_out_date.isoformat()} è già passata. "
            f"Inserisci una data a partire da {today.isoformat()}."
        )

    if check_out_date <= check_in_date:
        return (
            "⚠️ Le date dell'hotel non sono valide. "
            "La data di check-out deve essere successiva "
            "alla data di check-in."
        )

    # ========================================================
    # CONTROLLO BAMBINI
    # ========================================================

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

        for age in children_ages:

            if age < 0 or age > 17:
                return (
                    f"⚠️ L'età del bambino ({age}) non è valida. "
                    "L'età deve essere compresa tra 0 e 17 anni."
                )

    else:
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
    # LOG RICERCA
    # ========================================================

    print("=" * 80)
    print("HOTELS FINDER")
    print("=" * 80)

    print(f"Destinazione: {q}")
    print(f"Check-in: {check_in_date}")
    print(f"Check-out: {check_out_date}")
    print(f"Adulti: {adults}")
    print(f"Bambini: {children}")
    print(f"Classe hotel: {hotel_class}")

    # ========================================================
    # RICERCA SERPAPI
    # ========================================================

    try:

        search = GoogleSearch(query_params)

        results = search.get_dict()

        print(
            "SerpAPI response keys:",
            list(results.keys())
        )

        # ====================================================
        # ERRORE SERPAPI
        # ====================================================

        if "error" in results:

            error_message = results.get(
                "error",
                "Errore sconosciuto"
            )

            print(
                "SERPAPI HOTELS ERROR:",
                error_message
            )

            return (
                "⚠️ Non è stato possibile trovare hotel "
                "per i parametri indicati. "
                "Prova a modificare le date, la destinazione "
                "o i criteri di ricerca."
            )

        # ====================================================
        # RECUPERO PROPRIETÀ
        # ====================================================

        properties = results.get("properties", [])

        if not isinstance(properties, list):
            properties = []

        if not properties:

            print(
                "Nessun hotel trovato."
            )

            return (
                "⚠️ Non sono stati trovati hotel disponibili "
                "per le date e i parametri indicati."
            )

        # ====================================================
        # FILTRAGGIO HOTEL
        # ====================================================

        hotels = []

        for hotel in properties[:5]:

            if not isinstance(hotel, dict):
                continue

            # -----------------------------------------------
            # NOME
            # -----------------------------------------------

            name = hotel.get("name")

            if not name:
                continue

            # -----------------------------------------------
            # RATING
            # -----------------------------------------------

            rating = hotel.get("overall_rating")

            # -----------------------------------------------
            # RECENSIONI
            # -----------------------------------------------

            reviews = hotel.get("reviews")

            # -----------------------------------------------
            # CLASSE HOTEL
            # -----------------------------------------------

            hotel_class_value = hotel.get("hotel_class")

            # -----------------------------------------------
            # INDIRIZZO
            # -----------------------------------------------

            address = hotel.get("address")

            # -----------------------------------------------
            # DESCRIZIONE
            # -----------------------------------------------

            description = hotel.get("description")

            # -----------------------------------------------
            # PREZZO PER NOTTE
            # -----------------------------------------------

            rate_per_night = hotel.get(
                "rate_per_night",
                {}
            )

            if not isinstance(rate_per_night, dict):
                rate_per_night = {}

            price_per_night = rate_per_night.get(
                "lowest"
            )

            # -----------------------------------------------
            # PREZZO TOTALE
            # -----------------------------------------------

            total_rate = hotel.get(
                "total_rate",
                {}
            )

            if not isinstance(total_rate, dict):
                total_rate = {}

            total_price = total_rate.get(
                "lowest"
            )

            # -----------------------------------------------
            # LINK
            # -----------------------------------------------

            link = hotel.get("link")

            # -----------------------------------------------
            # COSTRUZIONE RISULTATO
            # -----------------------------------------------

            filtered_hotel = {
                "name": name,
                "rating": rating,
                "reviews": reviews,
                "hotel_class": hotel_class_value,
                "address": address,
                "price_per_night": price_per_night,
                "total_price": total_price,
                "description": description,
                "link": link,
            }

            # -----------------------------------------------
            # RIMOZIONE CAMPI VUOTI
            # -----------------------------------------------

            filtered_hotel = {
                key: value
                for key, value in filtered_hotel.items()
                if value is not None
            }

            hotels.append(filtered_hotel)

        # ====================================================
        # CONTROLLO RISULTATI FILTRATI
        # ====================================================

        if not hotels:

            print(
                "Gli hotel trovati da SerpAPI "
                "non contengono dati utilizzabili."
            )

            return (
                "⚠️ Sono stati trovati risultati, "
                "ma non è stato possibile recuperare "
                "informazioni utili sugli hotel."
            )

        # ====================================================
        # LOG RISULTATI
        # ====================================================

        print(
            f"Trovati {len(hotels)} hotel utilizzabili."
        )

        for index, hotel in enumerate(
            hotels,
            start=1
        ):

            print(
                f"{index}. {hotel.get('name')}"
            )

            if hotel.get("rating") is not None:
                print(
                    f"   Rating: {hotel.get('rating')}"
                )

            if hotel.get("price_per_night") is not None:
                print(
                    f"   Prezzo/notte: "
                    f"{hotel.get('price_per_night')}"
                )

            if hotel.get("total_price") is not None:
                print(
                    f"   Totale: "
                    f"{hotel.get('total_price')}"
                )

        print("=" * 80)

        # ====================================================
        # OUTPUT DEL TOOL
        # ====================================================

        return hotels

    # ========================================================
    # GESTIONE ERRORI
    # ========================================================

    except Exception as e:

        print("=" * 80)
        print("HOTELS FINDER ERROR")
        print("=" * 80)

        print(
            repr(e)
        )

        print("=" * 80)

        return (
            "⚠️ Si è verificato un errore durante "
            "la ricerca degli hotel."
        )