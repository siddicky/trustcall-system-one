"""Schemas and texts shared by the evals, the live smoke tests and the replay tests.

The README people conversation, the contact note, the order email and the
colleague announcement are the examples' scenarios, copied here (not imported) so
the cassettes recorded from them stay valid when an example is edited.
"""

# ruff: noqa: UP045 - the schemas deliberately spell Optional

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from trustcall_system_one import Many

__all__ = [
    "CONTACT_EXISTING",
    "CONTACT_NOTE",
    "CONVERSATION",
    "EXISTING_PEOPLE",
    "INVOICE_TEXT",
    "ORDER_EMAIL",
    "ORDER_EXISTING",
    "PASSAGE",
    "Bill",
    "Colleague",
    "Contact",
    "Invoice",
    "LineItem",
    "Member",
    "Order",
    "Person",
    "TotalOnly",
]

CONVERSATION = """\
Me: I ran into Emma Thompson at the park yesterday. She was walking her new puppy, a golden retriever named Sunny. She mentioned she got promoted to Senior Marketing Manager last month.
Friend: That's great news for Emma! How's she enjoying the new role?
Me: She seems to be thriving. Oh, and did you know she's taken up rock climbing? She invited me to join her at the climbing gym sometime.
Friend: Wow, rock climbing? That's quite a change from hiking. Speaking of friends, have you heard from Michael Chen recently?
Me: Actually, yes. We had a video call last week. He's switched jobs and is now working as a Data Scientist at a startup. He's also mentioned he's thinking of going vegan.
Friend: That's a big change for Michael! Both career and diet-wise. How about your neighbor, Sarah? Is she still teaching?
Me: Sarah's doing well. Her kids are growing up fast - her oldest just started middle school. She's still teaching, but now she's focusing on special education. She's really passionate about it.
Friend: That's wonderful. Oh, before I forget, I wanted to introduce you to my cousin who just moved to town. Her name is Olivia Davis, she's a 27-year-old graphic designer. She's looking to meet new people and expand her social circle. I thought you two might get along well.
Me: That sounds great! I'd love to meet her. Maybe we could all get together for coffee next week?
Friend: Perfect! I'll set it up. Olivia loves art and is always sketching in her free time. She also volunteers at the local animal shelter on weekends."""  # noqa: E501


class Person(BaseModel):
    """Someone the speaker knows or is being introduced to."""

    name: str = Field(description="full name")
    relationship: str = Field(description="how the person relates to the speaker")
    notes: list[str] = Field(description="things worth remembering about the person")


EXISTING_PEOPLE = [
    (
        "0",
        "Person",
        {
            "name": "Emma Thompson",
            "relationship": "College friend",
            "notes": ["Loves hiking", "Works in marketing", "Has a dog named Max"],
        },
    ),
    (
        "1",
        "Person",
        {
            "name": "Michael Chen",
            "relationship": "Coworker",
            "notes": ["Great at problem-solving", "Vegetarian", "Plays guitar"],
        },
    ),
    (
        "2",
        "Person",
        {
            "name": "Sarah Johnson",
            "relationship": "Neighbor",
            "notes": ["Has two kids", "Loves gardening", "Makes amazing cookies"],
        },
    ),
]


class Member(BaseModel):
    """A person in an address book."""

    name: str = Field(description="full name")
    city: Optional[str] = Field(None, description="the city the person lives in")
    phone: Optional[str] = Field(None, description="phone number")
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering about the person"
    )


class Colleague(BaseModel):
    """A colleague named in an announcement."""

    name: str = Field(description="full name")
    city: Optional[str] = Field(
        None, description="city of the office the person works in"
    )
    role: Optional[str] = Field(None, description="the person's job role")


PASSAGE = (
    "Priya Raman is moving to our Lisbon office next month. "
    "Tom Becker joins the same week as finance lead."
)


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


CONTACT_NOTE = (
    "Quick note after catching up with Dana: she moved to Berlin last month and "
    "started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. "
    "She also asked to be introduced to our logistics team."
)
CONTACT_EXISTING = {
    "name": "Dana Whitfield",
    "email": "dana.whitfield@fabrikam.example",
    "phone": "+1 415 555 0134",
    "city": "Austin",
    "company": "Fabrikam",
    "title": "Operations Manager",
    "notes": ["Prefers email over phone", "Met at the 2023 logistics summit"],
}


class Invoice(BaseModel):
    """An invoice."""

    invoice_number: str
    total: float = Field(description="Amount due including shipping, in USD")
    contacts: list[str]


class TotalOnly(BaseModel):
    """An invoice."""

    total: float = Field(description="Amount due including shipping, in USD")


class Bill(BaseModel):
    """An invoice with its number and total."""

    invoice_number: str
    total: float = Field(description="Amount due including shipping, in USD")


INVOICE_TEXT = (
    "Invoice INV-2291 from Northwind Supply, issued 2026-09-02, net 30. 4 pallets at "
    "$250.00 each, shipping $80.00. Contact: Dana Ruiz, Priya Shah."
)


class LineItem(BaseModel):
    """One product line on an order."""

    name: str = Field(description="product name")
    quantity: int = Field(ge=1, description="number of units ordered")
    sku: Optional[str] = Field(None, description="product code")


class Order(BaseModel):
    """A customer order."""

    order_id: str
    customer: str
    items: Annotated[list[LineItem], Many(identity="name")]


ORDER_EMAIL = (
    "Hi Sam, it's Ines about order SO-10442. Please add two more of the blue mugs "
    "and drop the quantity of the tea towels to 1. Everything else stays as it is. "
    "Thanks!"
)
ORDER_EXISTING = {
    "order_id": "SO-10442",
    "customer": "Ines Romero",
    "items": [
        {"name": "Steel kettle", "quantity": 1, "sku": "KTL-STL"},
        {"name": "Tea towels", "quantity": 4, "sku": "TWL-LIN"},
        {"name": "Wooden spoon set", "quantity": 2, "sku": "SPN-WD"},
    ],
}
