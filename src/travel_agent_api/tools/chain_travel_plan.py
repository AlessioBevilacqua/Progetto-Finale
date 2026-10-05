from datetime import date
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================
# INPUT DEL TOOL
# ============================================================

class TravelPlanInput(BaseModel):

    start_date: date = Field(
        description="The start date of the trip (YYYY-MM-DD)."
    )

    end_date: date = Field(
        description="The end date of the trip (YYYY-MM-DD)."
    )

    destination: str = Field(
        description="The destination of the trip."
    )

    adults: Optional[int] = Field(
        default=1,
        ge=1,
        description="The number of adults. Defaults to 1."
    )

    children: Optional[int] = Field(
        default=0,
        ge=0,
        description="The number of children. Defaults to 0."
    )

    children_ages: Optional[list[int]] = Field(
        default=None,
        description=(
            "The ages of all children. "
            "The number of ages must match the number of children. "
            "For example, [8, 10] for two children aged 8 and 10."
        )
    )

    travel_style: str = Field(
        description=(
            "The style of travel. Examples: adventure, relax, "
            "culture, backpacking, luxury, family-friendly."
        )
    )

    budget: Optional[int] = Field(
        default=None,
        ge=0,
        description="The total budget for the trip."
    )

    activities: str = Field(
        description=(
            "The preferred activities. Examples: culture, nature, "
            "food, shopping."
        )
    )

    food_restriction: str = Field(
        description=(
            "Any food restrictions. Examples: vegetarian, "
            "gluten-free, none."
        )
    )


# ============================================================
# OUTPUT DEL SINGOLO GIORNO
# ============================================================

class TravelDayOutput(BaseModel):

    morning: str = Field(
        description="Activities planned for the morning."
    )

    afternoon: str = Field(
        description="Activities planned for the afternoon."
    )

    evening: str = Field(
        description="Activities planned for the evening."
    )


# ============================================================
# OUTPUT COMPLETO DEL VIAGGIO
# ============================================================

class TravelPlanOutput(BaseModel):

    travel_plan: list[TravelDayOutput]


# ============================================================
# TOOL
# ============================================================

@tool(args_schema=TravelPlanInput)
def chain_travel_plan(
    start_date: date,
    end_date: date,
    destination: str,
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
    children_ages: Optional[list[int]] = None,
    travel_style: str = "",
    budget: Optional[int] = None,
    activities: str = "",
    food_restriction: str = "",
) -> str | TravelPlanOutput:
    """
    Creates a detailed day-by-day travel itinerary.
    """

    # ========================================================
    # INIZIO TOOL
    # ========================================================

    print("=" * 80)
    print("chain_travel_plan")
    print("=" * 80)

    # ========================================================
    # NORMALIZZAZIONE
    # ========================================================

    if adults is None:
        adults = 1

    if children is None:
        children = 0

    # ========================================================
    # DEBUG PARAMETRI
    # ========================================================

    print("-" * 80)
    print("TRAVEL PLAN PARAMETERS")
    print(f"Destination:       {destination}")
    print(f"Start date:        {start_date}")
    print(f"End date:          {end_date}")
    print(f"Adults:            {adults}")
    print(f"Children:          {children}")
    print(f"Children ages:     {children_ages}")
    print(f"Travel style:      {travel_style}")
    print(f"Budget:            €{budget if budget is not None else 'not specified'}")
    print(f"Activities:        {activities}")
    print(f"Food restrictions: {food_restriction}")
    print("-" * 80)

    # ========================================================
    # CONTROLLO DELLE DATE
    # ========================================================

    today = date.today()

    # --------------------------------------------------------
    # DATA DI INIZIO NEL PASSATO
    # --------------------------------------------------------

    if start_date < today:

        error_message = (
            "⚠️ Le date del viaggio non sono valide. "
            f"La data di inizio ({start_date}) è già passata. "
            f"Inserisci una data di partenza a partire da {today}."
        )

        print("TRAVEL PLAN ERROR:", error_message)
        print("=" * 80)

        return error_message

    # --------------------------------------------------------
    # DATA DI FINE NEL PASSATO
    # --------------------------------------------------------

    if end_date < today:

        error_message = (
            "⚠️ Le date del viaggio non sono valide. "
            f"La data di fine ({end_date}) è già passata. "
            f"Inserisci una data di fine a partire da {today}."
        )

        print("TRAVEL PLAN ERROR:", error_message)
        print("=" * 80)

        return error_message

    # --------------------------------------------------------
    # DATA FINE PRIMA DELLA DATA INIZIO
    # --------------------------------------------------------

    if end_date < start_date:

        error_message = (
            "⚠️ Le date del viaggio non sono valide. "
            "La data di fine non può essere precedente "
            "alla data di inizio."
        )

        print("TRAVEL PLAN ERROR:", error_message)
        print("=" * 80)

        return error_message

    # ========================================================
    # CONTROLLO BAMBINI
    # ========================================================

    if children > 0:

        if not children_ages:

            error_message = (
                "⚠️ Sono stati indicati dei bambini, "
                "ma non sono state specificate le loro età."
            )

            print("TRAVEL PLAN ERROR:", error_message)
            print("=" * 80)

            return error_message

        if len(children_ages) != children:

            error_message = (
                f"⚠️ Sono stati indicati {children} bambini, "
                f"ma sono state specificate "
                f"{len(children_ages)} età. "
                "Il numero delle età deve corrispondere "
                "al numero dei bambini."
            )

            print("TRAVEL PLAN ERROR:", error_message)
            print("=" * 80)

            return error_message

        for age in children_ages:

            if age < 0 or age > 17:

                error_message = (
                    f"⚠️ L'età del bambino ({age}) non è valida. "
                    "L'età deve essere compresa tra 0 e 17 anni."
                )

                print("TRAVEL PLAN ERROR:", error_message)
                print("=" * 80)

                return error_message

    else:

        children_ages = None

    # ========================================================
    # GENERAZIONE DEL PIANO
    # ========================================================

    try:

        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7
        )

        structured_model = model.with_structured_output(
            TravelPlanOutput
        )

        # ====================================================
        # FORMATTAZIONE ETÀ BAMBINI
        # ====================================================

        if children_ages:

            children_information = (
                f"{children} children, "
                f"aged {', '.join(map(str, children_ages))}"
            )

        else:

            children_information = str(children)

        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are an expert travel planner.

Create a detailed day-by-day itinerary using the following
travel information.

Destination:
{destination}

Start date:
{start_date}

End date:
{end_date}

Adults:
{adults}

Children:
{children_information}

Travel style:
{travel_style}

Budget:
€{budget if budget is not None else "not specified"}

Preferred activities:
{activities}

Food restrictions:
{food_restriction}

Requirements:

- Create one itinerary entry for each day of the trip.
- Divide every day into morning, afternoon and evening.
- Adapt the activities to the destination.
- Respect the requested travel style.
- Consider the available budget.
- Consider the number of adults and children.
- Consider the exact ages of the children.
- Choose activities that are appropriate for their ages.
- Consider whether activities are suitable for a family.
- Consider the requested activities.
- Respect all food restrictions.
- If one or more travelers have celiac disease or another
  food restriction, recommend appropriate food options.
- Avoid unrealistic schedules.
- Avoid scheduling too many activities in a single day.
- Use realistic travel times between activities.
- Include appropriate breaks for children.
- Do not invent impossible or contradictory information.
- Do not invent real-time prices or availability.
- Do not claim that flights, hotels or attractions are booked.
- The itinerary should be practical and realistic.
"""

        # ====================================================
        # CHIAMATA MODELLO
        # ====================================================

        result = structured_model.invoke(prompt)

        # ====================================================
        # RISULTATO
        # ====================================================

        print("-" * 80)
        print("TRAVEL PLAN GENERATED SUCCESSFULLY")
        print("-" * 80)

        if isinstance(result, TravelPlanOutput):

            print(
                f"Giorni generati: {len(result.travel_plan)}"
            )

            print()

            for index, day in enumerate(
                result.travel_plan,
                start=1
            ):

                print(f"Giorno {index}")
                print(f"  Mattina:     {day.morning}")
                print(f"  Pomeriggio:  {day.afternoon}")
                print(f"  Sera:        {day.evening}")
                print()

        else:

            print(result)

        print("=" * 80)

        return result

    # ========================================================
    # GESTIONE ERRORI
    # ========================================================

    except Exception as e:

        print("=" * 80)
        print("TRAVEL PLAN ERROR")
        print("=" * 80)
        print(repr(e))
        print("=" * 80)

        raise