import sys

def load_set(filepath):
    data = {}
    curr_q = -1
    f = open(filepath, 'r')
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('q #'):
            curr_q = int(line.split()[2])
        elif line.startswith('c #'):
            parts = line.split()
            if len(parts) > 2:
                vals = set([int(x) for x in parts[2:]])
            else:
                vals = set()
            data[curr_q] = vals
        elif ':' in line:
            parts = line.split(':', 1)
            q_id = int(parts[0].strip())
            matches_str = parts[1].strip()
            if matches_str:
                vals = set([int(x.strip()) for x in matches_str.split(',') if x.strip()])
            else:
                vals = set()
            data[q_id] = vals
    f.close()
    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_fast.py <truth_file> <candidate_file>")
        sys.exit(1)

    truth_path = sys.argv[1]
    cand_path = sys.argv[2]
    
    truth = load_set(truth_path)
    cands = load_set(cand_path)
    
    print("-" * 60)
    print(f"{'Query':<6} | {'True':<12} | {'Yours':<12} | {'Score':<8} | {'Status'}")
    print("-" * 60)
    
    total = 0
    count = 0
    
    for q_id in sorted(truth.keys()):
        R_q = truth[q_id]
        C_q = cands.get(q_id, set())
        
        missed = R_q - C_q
        
        if missed:
            status = "FAIL"
            score = 0.0
        else:
            status = "PASS"
            if len(C_q) > 0:
                score = len(R_q) / len(C_q)
            else:
                score = 0.0
        
        total += score
        count += 1
        
        print(f"{q_id:<6} | {len(R_q):<12} | {len(C_q):<12} | {score:.4f}   | {status}")
        if missed:
            print(f"   ERROR: Missed {list(missed)[:3]}")

    if count > 0:
        avg = total / count
        print("-" * 60)
        print(f"AVG SCORE: {avg:.4f}")
    else:
        print("No queries found")