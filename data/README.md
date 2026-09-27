# RetailRocket Dataset

This project uses the RetailRocket e-commerce dataset.

## Main Dataset

The main file used by this project is:

`events.csv`

Important columns:

- `timestamp` - Event timestamp
- `visitorid` - Anonymous visitor identifier
- `event` - User interaction type
- `itemid` - Item identifier
- `transactionid` - Transaction identifier

## Event Types

The relevant event types are:

- `view`
- `addtocart`
- `transaction`

Transaction events are used to construct purchase labels but are not used as model inputs.

## Dataset Setup

Download the RetailRocket dataset and place:

`events.csv`

inside:

`data/raw/`

Expected path:

`data/raw/events.csv`

The raw dataset must not be committed to GitHub.

## Session Definition

Events are grouped into browsing sessions.

Events are ordered chronologically for each visitor.

A new session starts when the inactivity gap between consecutive events from the same visitor is greater than 30 minutes.

## Target

Each valid session receives a binary target:

- `0` - No purchase
- `1` - Purchase

For purchase sessions, only behavioural events occurring before the first transaction are used as model input.

This prevents target leakage.