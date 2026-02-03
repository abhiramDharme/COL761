import sys
import numpy as np

def generate_candidates(db_features, query_features):
    candidates = []

    for qi in range(query_features.shape[0]):
        cand = []
        qv = query_features[qi]

        for gi in range(db_features.shape[0]):
            gv = db_features[gi]

            if np.all(qv <= gv):
                cand.append(gi)  # 0-based indexing

        candidates.append(cand)

    return candidates

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_candidates.py <db_features.npy> <query_features.npy> <output.dat>")
        sys.exit(1)
    
    db_features_path = sys.argv[1]
    query_features_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("Loading database features...")
    db_features = np.load(db_features_path)
    print(f"Database features shape: {db_features.shape}")
    
    print("Loading query features...")
    query_features = np.load(query_features_path)
    print(f"Query features shape: {query_features.shape}")
    
    print("Generating candidates...")
    candidates = generate_candidates(db_features, query_features)

    with open(output_path, "w") as f:
        for i, cand in enumerate(candidates):
            f.write(f"q # {i}\n")  # 0-based indexing
            f.write("c # " + " ".join(map(str, cand)) + "\n")
    
    print(f"Generated candidates for {len(candidates)} queries")
    for i in range(min(5, len(candidates))):
        print(f"  Query {i+1}: {len(candidates[i])} candidates")
    
    print(f"Saved candidates to {output_path}")

if __name__ == "__main__":
    main()