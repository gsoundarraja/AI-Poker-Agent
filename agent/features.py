# extract features defined in evaluation.py
STREET_INDEX = {"preflop": 0, "flop": 1, "turn":2,"river": 3, "showdown":3}

def find_seats(round_state, agent_uuid):
    agent_seat, opp_seat = None, None
    for seat in round_state["seats"]:
        if seat["uuid"] == agent_uuid:
            agent_seat = seat
        else:
            opp_seat = seat
    return agent_seat, opp_seat

# amt committed in street 
def street_commitments(round_state, agent_uuid):
    street = round_state["street"]
    hist = round_state.get("action_histories",{}).get(street, [])  or []
    agent_paid = 0
    opp_paid = 0
    # compute pot odds
    for entry in hist:
        paid = entry.get("paid", 0) or 0
        if entry.get("uuid") == agent_uuid:
            agent_paid += paid
        else:
            opp_paid += paid
    return agent_paid, opp_paid

# raised by fixed amnt from engine
def round_raise_increment(round_state):
    sb = round_state.get("small_blind_amount", 10)
    if STREET_INDEX[round_state["street"]] <= 1:
        return sb * 2
    return sb * 4

def pot_size(round_state):
    pot = round_state.get("pot", {})
    total = pot.get("main", {}).get("amount", 0)
    for side in pot.get("side", []) or []:
        total += side.get("amount", 0)
    return total

def pot_odds(to_call, pot):
    if to_call <= 0:
        return 0.0
    return float(to_call) /float(pot + to_call)

# +1 if after -1 if before
def position_indicator(round_state, agent_uuid):
    seats = round_state["seats"]
    agent_idx = 0
    for i, s in enumerate(seats):
        if s["uuid"] == agent_uuid:
            agent_idx = i
            break
    dealer = round_state.get("dealer_btn", 0)
    if round_state["street"] == "preflop":
        return -1 if agent_idx == dealer else 1
    else:
        return 1 if agent_idx == dealer else -1

# amount of stack already in pot
def pot_commitment(round_state, agent_uuid, initial_stack_fallback=10000):
    agent_seat, _ = find_seats(round_state, agent_uuid)
    stack = agent_seat.get("stack", initial_stack_fallback)
    committed = max(0, initial_stack_fallback - stack)
    return float(committed)/ float(initial_stack_fallback)

# put into dict
def extract_state_features(round_state, hole_card, agent_uuid, initial_stack):
    agent_seat, opp_seat = find_seats(round_state, agent_uuid)
    agent_paid, opp_paid =street_commitments(round_state, agent_uuid)
    to_call= max(0, opp_paid - agent_paid)
    pot = pot_size(round_state)
    return {
        "street": round_state["street"],
        "street_idx": STREET_INDEX[round_state["street"]],
        "pot": pot,
        "to_call": to_call,
        "raise_increment": round_raise_increment(round_state),
        "pot_odds": pot_odds(to_call, pot),
        "position": position_indicator(round_state, agent_uuid),
        "pot_commitment": pot_commitment(round_state, agent_uuid, initial_stack),
        "agent_stack": agent_seat.get("stack", initial_stack),
        "opp_stack": opp_seat.get("stack", initial_stack),
        "opp_uuid": opp_seat.get("uuid"),
        "small_blind": round_state.get("small_blind_amount", 10),
        "hole": hole_card,
        "community": round_state.get("community_card", [])
    }
