import random
random.seed(42)

CORE_ITEMS = list(range(0, 5))
PLATEAU_ITEMS = list(range(5, 20))
TAIL_ITEMS = list(range(20, 30))

seen = set()
generated = 0
unique = 0
plateau_full_count = 0
plateau_full_unique = 0

for _ in range(50000):
    generated += 1
    trans = []
    
    # Core
    for item in CORE_ITEMS:
        if random.random() < 0.95:
            trans.append(item)
    
    # Plateau
    full_plateau = random.random() < 0.55
    if full_plateau:
        trans.extend(PLATEAU_ITEMS)
        plateau_full_count += 1
    else:
        for item in PLATEAU_ITEMS:
            if random.random() < 0.33:
                trans.append(item)
    
    # Tail
    for item in TAIL_ITEMS:
        if random.random() < 0.07:
            trans.append(item)
    
    trans_tuple = tuple(sorted(trans))
    if trans_tuple not in seen:
        seen.add(trans_tuple)
        unique += 1
        if full_plateau:
            plateau_full_unique += 1
    
    if unique >= 15000:
        break

print(f"Generated: {generated}, Unique: {unique}")
print(f"Plateau full in generated: {plateau_full_count/generated*100:.1f}%")
print(f"Plateau full in unique: {plateau_full_unique/unique*100:.1f}%")
