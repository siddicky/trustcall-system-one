/**
 * Ready-made scenarios. The Contact update is the README's "Without the
 * library vs with the library" example; the Invoice and the People memory come
 * from `examples/01_invoice.py` and `examples/05_people_memory.py`.
 */
import { makeField, type FormField } from "./schema"
import type { Cardinality, JsonObject } from "./types"

export interface PresetSchema {
  name: string
  description: string
  cardinality: Cardinality
  fields: FormField[]
}

export interface PresetExisting {
  id: string
  schema: string
  record: JsonObject
}

export interface Preset {
  id: string
  label: string
  summary: string
  schemas: PresetSchema[]
  /** Title of the pasted document. */
  documentLabel: string
  text: string
  existing: PresetExisting[]
  enableInserts: boolean
  enableDeletes: boolean
}

// ---------------------------------------------------------------------------
// (a) Contact update, from the README
// ---------------------------------------------------------------------------

const CONTACT_NOTE =
  "Quick note after catching up with Dana: she moved to Berlin last month and " +
  "started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. " +
  "She also asked to be introduced to our logistics team."

const CONTACT_RECORD: JsonObject = {
  name: "Dana Whitfield",
  email: "dana.whitfield@fabrikam.example",
  phone: "+1 415 555 0134",
  city: "Austin",
  company: "Fabrikam",
  title: "Operations Manager",
  notes: ["Prefers email over phone", "Met at the 2023 logistics summit"],
}

function contactFields(): FormField[] {
  return [
    makeField({ name: "name" }),
    makeField({ name: "email" }),
    makeField({ name: "phone", required: false }),
    makeField({ name: "city", required: false }),
    makeField({ name: "company", required: false }),
    makeField({ name: "title", required: false }),
    makeField({
      name: "notes",
      type: "array",
      required: false,
      description: "things worth remembering",
    }),
  ]
}

// ---------------------------------------------------------------------------
// (b) Invoice insert, from examples/01_invoice.py
// ---------------------------------------------------------------------------

const INVOICE_EMAIL = `\
From: billing@brightwave-supplies.example
To: priya.nair@northwind-traders.example
Subject: Invoice INV-2024-0457 from Brightwave Supplies Ltd

Hi Priya,

Please find attached invoice INV-2024-0457 from Brightwave Supplies Ltd, issued on 14 March 2024 for the standing desks delivered last week.

Subtotal: $1,000.00
Sales tax (8%): $80.00
Total due: $1,080.00 USD

Payment terms are net 30 days (2% off if paid within 10 days). Your previous invoice, INV-2024-0412, was settled on 2 March, thank you. Please send any questions to Dana Whitlock on our accounts team.

Kind regards,
Brightwave Supplies Accounts`

function invoiceFields(): FormField[] {
  return [
    makeField({
      name: "invoice_number",
      description: "invoice number of this invoice",
    }),
    makeField({ name: "vendor", description: "company that issued the invoice" }),
    makeField({
      name: "issue_date",
      type: "date",
      description: "date the invoice was issued",
    }),
    makeField({
      name: "terms_days",
      type: "integer",
      description: "payment terms in days",
    }),
    makeField({
      name: "total",
      type: "number",
      description: "amount due on the invoice, tax included",
    }),
    makeField({
      name: "currency",
      type: "enum",
      enumValues: ["USD", "EUR", "GBP"],
    }),
    makeField({
      name: "paid",
      type: "boolean",
      description: "this invoice has already been paid",
    }),
    makeField({
      name: "po_number",
      required: false,
      description: "purchase order number",
    }),
    makeField({
      name: "approved_by",
      required: false,
      description:
        "person at the customer who approved paying the invoice",
    }),
  ]
}

// ---------------------------------------------------------------------------
// (c) People memory, from examples/05_people_memory.py
// ---------------------------------------------------------------------------

const CONVERSATION = `\
Me: I ran into Emma Thompson at the park yesterday. She was walking her new puppy, a golden retriever named Sunny. She mentioned she got promoted to Senior Marketing Manager last month.
Friend: That's great news for Emma! How's she enjoying the new role?
Me: She seems to be thriving. Oh, and did you know she's taken up rock climbing? She invited me to join her at the climbing gym sometime.
Friend: Wow, rock climbing? That's quite a change from hiking. Speaking of friends, have you heard from Michael Chen recently?
Me: Actually, yes. We had a video call last week. He's switched jobs and is now working as a Data Scientist at a startup. He's also mentioned he's thinking of going vegan.
Friend: That's a big change for Michael! Both career and diet-wise. How about your neighbor, Sarah? Is she still teaching?
Me: Sarah's doing well. Her kids are growing up fast - her oldest just started middle school. She's still teaching, but now she's focusing on special education. She's really passionate about it.
Friend: That's wonderful. Oh, before I forget, I wanted to introduce you to my cousin who just moved to town. Her name is Olivia Davis, she's a 27-year-old graphic designer. She's looking to meet new people and expand her social circle. I thought you two might get along well.
Me: That sounds great! I'd love to meet her. Maybe we could all get together for coffee next week?
Friend: Perfect! I'll set it up. Olivia loves art and is always sketching in her free time. She also volunteers at the local animal shelter on weekends.`

function personFields(): FormField[] {
  return [
    makeField({ name: "name", description: "full name" }),
    makeField({
      name: "relationship",
      description: "how the person relates to the speaker",
    }),
    makeField({
      name: "notes",
      type: "array",
      description: "things worth remembering about the person",
    }),
  ]
}

export const PRESETS: Preset[] = [
  {
    id: "contact",
    label: "Contact update",
    summary: "A call note updates one CRM record (crm-4711).",
    schemas: [
      {
        name: "Contact",
        description: "A contact in the CRM.",
        cardinality: "one",
        fields: contactFields(),
      },
    ],
    documentLabel: "contact-note.txt",
    text: CONTACT_NOTE,
    existing: [{ id: "crm-4711", schema: "Contact", record: CONTACT_RECORD }],
    enableInserts: false,
    enableDeletes: false,
  },
  {
    id: "invoice",
    label: "Invoice insert",
    summary: "An invoice email becomes a new typed record.",
    schemas: [
      {
        name: "Invoice",
        description: "An invoice received from a vendor.",
        cardinality: "one",
        fields: invoiceFields(),
      },
    ],
    documentLabel: "invoice-email.txt",
    text: INVOICE_EMAIL,
    existing: [],
    enableInserts: true,
    enableDeletes: false,
  },
  {
    id: "people",
    label: "People memory",
    summary: "A chat updates three people and introduces a fourth.",
    schemas: [
      {
        name: "Person",
        description: "Someone the speaker knows or is being introduced to.",
        cardinality: "many",
        fields: personFields(),
      },
    ],
    documentLabel: "conversation.txt",
    text: CONVERSATION,
    existing: [
      {
        id: "0",
        schema: "Person",
        record: {
          name: "Emma Thompson",
          relationship: "College friend",
          notes: ["Loves hiking", "Works in marketing", "Has a dog named Max"],
        },
      },
      {
        id: "1",
        schema: "Person",
        record: {
          name: "Michael Chen",
          relationship: "Coworker",
          notes: ["Great at problem-solving", "Vegetarian", "Plays guitar"],
        },
      },
      {
        id: "2",
        schema: "Person",
        record: {
          name: "Sarah Johnson",
          relationship: "Neighbor",
          notes: ["Has two kids", "Loves gardening", "Makes amazing cookies"],
        },
      },
    ],
    enableInserts: true,
    enableDeletes: false,
  },
]
