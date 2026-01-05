from flask import Flask, render_template, request, redirect, url_for
from openai_stuff import extract_tasting_notes, get_embeddings
import sys
from rarity_metrics import get_rarity_scores
from load_csv_notes import load_notes_from_csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///submissions.db'
db = SQLAlchemy(app)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    participant = db.Column(db.String(250), nullable=False)
    message = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_similarity_matrix(embeddings1, embeddings2):
    matrix = np.zeros((len(embeddings1), len(embeddings2)))
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            matrix[i, j] = cosine_similarity(emb1, emb2)
    return matrix

def optimal_matching(sim_matrix):
    # Since the algorithm minimizes total cost, we use negative similarities
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_similarity = sim_matrix[row_ind, col_ind].sum()
    matched_similarities = sim_matrix[row_ind, col_ind]

    return row_ind, col_ind, total_similarity, matched_similarities


def adjusted_similarity_with_threshold(notes1, notes2, total_similarity, matched_indices, matched_similarities,
                                       similarity_threshold=0.5):
    ''' calculate "soft" IOU. note that I am adding 1 to intersection and 2 to union (basically hallucinating
        a single match no matter what-- to make sure the weight is non-zero '''
    row_ind, col_ind = matched_indices
    # Count matches with similarity above the threshold
    intersection_count = sum(sim > similarity_threshold for sim in matched_similarities) + 1

    # Total unique notes (union)
    total_notes = len(notes1) + len(notes2) + 2
    union_count = total_notes - intersection_count

    # Compute IoU
    iou = intersection_count / union_count if union_count != 0 else 0

    # Adjust the total similarity score
    adjusted_score = total_similarity * iou
    #print(f"total sim {total_similarity}, iou {iou} = intesection {intersection_count}/{union_count}, adjusted score {adjusted_score}")
    return adjusted_score




def display_matching(notes1, notes2, row_ind, col_ind, sim_matrix, rarity1, rarity2, closest_wheel1, closest_wheel2, similarity_threshold=0.5):
    matched_pairs = []
    unmatched_notes1 = set(range(len(notes1)))
    unmatched_notes2 = set(range(len(notes2)))

    for idx, (i, j) in enumerate(zip(row_ind, col_ind)):
        similarity = sim_matrix[i, j]
        is_above_threshold = similarity > similarity_threshold
        matched_pairs.append((notes1[i], notes2[j], similarity, is_above_threshold,
                              rarity1[i], rarity2[j], closest_wheel1[i], closest_wheel2[j]))
        unmatched_notes1.discard(i)
        unmatched_notes2.discard(j)

    # Display the matches
    print("Matched Notes:")
    for note1, note2, sim, is_in_intersection, r1, r2, w1, w2 in matched_pairs:
        status = "Included in Intersection" if is_in_intersection else "Not Included in Intersection"
        print(f" - '{note1}' matched with '{note2}' (Similarity: {sim:.4f}) [{status}]")

        if is_in_intersection and (r1 > 0.85 and r2 > 0.85):
            print()
            print(f"  *  RARE MATCH! '{note1}' closest to wheel note '{w1}' (rarity={r1:.4f}) and '{note2}' closest to '{w2}' (rarity={r2:.4f})")
            print()

    # Display unmatched notes
    if unmatched_notes1:
        print("\nUnmatched Notes from First Participant:")
        for idx in unmatched_notes1:
            print(f" - '{notes1[idx]}'")
    else:
        print("\nAll notes from the first participant were matched.")

    if unmatched_notes2:
        print("\nUnmatched Notes from Second Participant:")
        for idx in unmatched_notes2:
            print(f" - '{notes2[idx]}'")
    else:
        print("\nAll notes from the second participant were matched.")


def process_tasting_notes(participants_messages, official_message, sim_threshold=0.5, verbose=True):
    # Extract participant tasting notes
    participants_notes = {}
    participants_embeddings = {}
    participants_rarity = {}

    for participant, message in participants_messages.items():
        notes = extract_tasting_notes(message).notes
        embeddings = get_embeddings(notes)

        rarity_scores, closest_wheel_notes, dist_to_wheel = get_rarity_scores(notes, embeddings)

        participants_notes[participant] = notes
        participants_embeddings[participant] = embeddings
        participants_rarity[participant] = [rarity_scores, closest_wheel_notes, dist_to_wheel]

    # Extract official notes
    official_notes = extract_tasting_notes(official_message).notes
    official_embeddings = get_embeddings(official_notes)
    official_rarity_scores, official_closest_wheel_notes, official_dist_to_wheel = get_rarity_scores(official_notes,
                                                                                                    official_embeddings)

    # Compute similarity to official notes
    scores_to_official = {}
    for participant in participants_notes:
        notes1 = participants_notes[participant]
        embeddings1 = participants_embeddings[participant]
        rarity_scores, closest_wheel_notes, dist_to_wheel = get_rarity_scores(notes1, embeddings1)

        notes2 = official_notes
        embeddings2 = official_embeddings

        sim_matrix = compute_similarity_matrix(embeddings1, embeddings2)
        row_ind, col_ind, total_similarity, matched_similarities = optimal_matching(sim_matrix)

        adjusted_score = adjusted_similarity_with_threshold(
            notes1, notes2, total_similarity, (row_ind, col_ind), matched_similarities, similarity_threshold=sim_threshold)

        scores_to_official[participant] = adjusted_score

        # Display matching for debugging
        if verbose:
            print(f"\nMatches between {participant} and Official Notes:")
            display_matching(notes1, notes2, row_ind, col_ind, sim_matrix,
                             rarity_scores, official_rarity_scores,
                             closest_wheel_notes, official_closest_wheel_notes,
                             similarity_threshold=sim_threshold)

    # Determine the closest participant to the official notes
    winner = max(scores_to_official, key=scores_to_official.get)
    if verbose: print(f"The participant closest to the official notes is {winner}.")

    # Compute pairwise similarities between participants
    participants = list(participants_notes.keys())
    pairwise_scores = {}
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            p1, p2 = participants[i], participants[j]
            notes1, embeddings1 = participants_notes[p1], participants_embeddings[p1]
            rarity_scores1, closest_wheel_notes1, dist_to_wheel1 = participants_rarity[p1]
            notes2, embeddings2 = participants_notes[p2], participants_embeddings[p2]
            rarity_scores2, closest_wheel_notes2, dist_to_wheel2 = participants_rarity[p2]

            sim_matrix = compute_similarity_matrix(embeddings1, embeddings2)
            row_ind, col_ind, total_similarity, matched_similarities = optimal_matching(sim_matrix)

            adjusted_score = adjusted_similarity_with_threshold(
                notes1, notes2, total_similarity, (row_ind, col_ind), matched_similarities, similarity_threshold=sim_threshold)
            pairwise_scores[(p1, p2)] = adjusted_score

            # Display matching for debugging
            if verbose:
                print(f"\nMatches between {p1} and {p2}:")
                #display_matching(notes1, notes2, row_ind, col_ind, sim_matrix, similarity_threshold=sim_threshold)
                display_matching(notes1, notes2, row_ind, col_ind, sim_matrix,
                                 rarity_scores1, rarity_scores2,
                                 closest_wheel_notes1, closest_wheel_notes2,
                                 similarity_threshold=sim_threshold)

    # Determine the pair with the highest agreement
    most_agreed_pair = max(pairwise_scores, key=pairwise_scores.get)
    if verbose: print(f"The pair with the most agreement is {most_agreed_pair[0]} and {most_agreed_pair[1]}.")

    if verbose: print("Rarest note award")

    # Initialize a list to collect all notes with their metadata
    all_notes = []

    for p1 in participants:
        notes = participants_notes[p1]
        rarity_scores, closest_wheel_notes, dist_to_wheel = participants_rarity[p1]

        for i in range(len(notes)):
            note = notes[i]
            rarity_score = rarity_scores[i]
            closest_note = closest_wheel_notes[i]
            distance = dist_to_wheel[i]

            # Collect all relevant data into a dictionary
            all_notes.append({
                'participant': p1,
                'note': note,
                'rarity_score': rarity_score,
                'closest_wheel_note': closest_note,
                'distance_to_wheel_note': distance
            })

    # Sort the list of notes descending by rarity score
    sorted_notes = sorted(all_notes, key=lambda x: x['rarity_score'], reverse=True)

    # Print the top 3 rarest notes along with their metadata
    top_n = 3
    if verbose:
        print(f"Top {top_n} Rarest Notes:")
        for i in range(min(top_n, len(sorted_notes))):
            entry = sorted_notes[i]
            print(f"{i + 1}. Participant: {entry['participant']}")
            print(f"   Note: '{entry['note']}'")
            print(f"   Rarity Score: {entry['rarity_score']:.4f}")
            print(f"   Closest Wheel Note: '{entry['closest_wheel_note']}'")
            print(f"   Distance to Wheel Note: {entry['distance_to_wheel_note']:.4f}\n")

    return scores_to_official, pairwise_scores

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Store submissions temporarily in a global variable (for demo purposes)
submissions = {}
official_message = "The coffee features tasting notes of dark chocolate, orange zest, and oolong tea."

@app.route('/submit_notes', methods=['POST'])
def submit_notes():
    participant = request.form['participant']
    message = request.form['message']
    submission = Submission(participant=participant, message=message)
    db.session.add(submission)
    db.session.commit()
    return redirect(url_for('results'))

@app.route('/results', methods=['GET'])
def results():
    submissions = {s.participant: s.message for s in Submission.query.all()}
    if not submissions:
        return "No submissions yet."

    # Process the tasting notes
    scores_to_official, pairwise_scores = process_tasting_notes(submissions, official_message)

    return render_template('results.html', scores_to_official=scores_to_official, pairwise_scores=pairwise_scores)

def run_app():
    app.run(debug=True)

def run_local_demo(day, verbose=True):

    # Load notes from CSV (Shervin and Caitlin only)
    db = load_notes_from_csv(participants=['shervin', 'caitlin'])

    # Official tasting notes
    official_message = db[day]['Onyx']

    # everyone but onyx
    participants_messages = dict((k,v) for k, v in db[day].items() if k != 'Onyx')

    # Process the tasting notes
    scores_to_official, pairwise_scores = process_tasting_notes(participants_messages, official_message, verbose=verbose)

    # Display the results
    print("\nScores to Official Notes:")
    for i, (participant, score) in enumerate(sorted(scores_to_official.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i+1}. {participant}: {score:.4f}")

    print("\nPairwise Agreement Scores:")
    for i, (pair, score) in enumerate(sorted(pairwise_scores.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i+1}. {pair[0]} and {pair[1]}: {score:.4f}")

if __name__ == '__main__':

    if len(sys.argv) > 1:
        day = int(sys.argv[1])  # Convert the argument to an integer
    else:
        day = 1

    # just running all the functions with manually input notes from people's signal messages
    # so, no sqlite or anything, just all the day's data in a big dictionary
    run_local_demo(day=day, verbose=True)

    #for day in range(1, 9):
    #    print("Day", day)
    #    run_local_demo(day=day, verbose=False)

    #for day in range(9, 12):
    #    print("Day", day)
    #    run_local_demo(day=day, verbose=False)

    # run_app() # for running as a Flask web app with sqlite backend