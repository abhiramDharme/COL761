source venv/bin/activate

# Usage: ./generate_candidates.sh <db_vecs> <q_vecs> <output> 
python3 candidate_gen.py "$1" "$2" "$3"