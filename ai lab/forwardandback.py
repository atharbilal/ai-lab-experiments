class Rule:
    def __init__(self, premises, conclusion):
        self.premises = set(premises)
        self.conclusion = conclusion

    def __repr__(self):
        return f"IF {' AND '.join(self.premises)} THEN {self.conclusion}"

def forward_chaining(rules, initial_facts, goal):
    """
    Starts with known facts and repeatedly applies rules to derive 
    new facts until the goal is reached or no new facts can be inferred.
    """
    known_facts = set(initial_facts)
    inferred_facts = []

    print("\n--- Forward Chaining Trace ---")
    print(f"Initial Facts: {sorted(list(known_facts))}")
    print(f"Goal: {goal}\n")

    added_new_fact = True
    step = 1

    while added_new_fact:
        added_new_fact = False
        
        if goal in known_facts:
            print(f" Goal '{goal}' already reached!")
            break

        for rule in rules:
            if rule.premises.issubset(known_facts) and rule.conclusion not in known_facts:
                known_facts.add(rule.conclusion)
                inferred_facts.append(rule.conclusion)
                added_new_fact = True
                print(f"Step {step}: Applied rule [{rule}] -> Inferred '{rule.conclusion}'")
                step += 1
                
                if rule.conclusion == goal:
                    break

    print("\n--- Forward Chaining Result ---")
    if goal in known_facts:
        print(f"SUCCESS: Goal '{goal}' was successfully derived!")
        print(f"All Known Facts: {sorted(list(known_facts))}")
        return True
    else:
        print(f"FAILURE: Goal '{goal}' could not be derived.")
        return False

def backward_chaining(rules, known_facts, goal, visited=None):
    """
    Starts with the goal and works backward to check if known facts 
    and rules support it using recursive depth-first search.
    """
    if visited is None:
        visited = set()

    if goal in known_facts:
        print(f"Fact '{goal}' is directly known.")
        return True

    if goal in visited:
        return False
    visited.add(goal)

    print(f"Attempting to prove goal: '{goal}'")

    matching_rules = [r for r in rules if r.conclusion == goal]

    if not matching_rules:
        print(f"No rules found to prove '{goal}'.")
        return False

    for rule in matching_rules:
        print(f"To prove '{goal}', checking rule [{rule}]...")

        all_premises_proven = True
        for premise in rule.premises:
            print(f"  Sub-goal: Need to prove '{premise}' for rule [{rule}]")
            if not backward_chaining(rules, known_facts, premise, visited.copy()):
                all_premises_proven = False
                break  

        if all_premises_proven:
            print(f"PROVEN: Goal '{goal}' is supported by rule [{rule}].")
            return True

    print(f"FAILED: Goal '{goal}' could not be proven.")
    return False

if __name__ == "__main__":

    kb_rules = [
        Rule(['A', 'B'], 'C'),  # Rule 1
        Rule(['C'], 'D'),       # Rule 2
        Rule(['D'], 'E')        # Rule 3
    ]

    facts = {'A', 'B'}

    print("==========================================")
    print("        PART A: FORWARD CHAINING          ")
    print("==========================================")
    fc_success = forward_chaining(rules=kb_rules, initial_facts=facts, goal='D')

    print("\n==========================================")
    print("        PART B: BACKWARD CHAINING         ")
    print("==========================================")
    target_goal = 'E'
    print(f"Initial Facts: {sorted(list(facts))}")
    print(f"Goal to prove: {target_goal}\n")
    
    print("--- Backward Chaining Trace ---")
    bc_success = backward_chaining(rules=kb_rules, known_facts=facts, goal=target_goal)

    print("\n--- Backward Chaining Result ---")
    if bc_success:
        print(f"SUCCESS: Goal '{target_goal}' is proven true!")
        print(f"Derived path: {target_goal} <- D <- C <- (A AND B)")
    else:
        print(f"FAILURE: Goal '{target_goal}' cannot be proven.")