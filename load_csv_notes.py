import pandas as pd

def load_notes_from_csv(csv_path='./data/onyx_advent_2025.csv', participants=None):
    """
    Load tasting notes from CSV file.

    Args:
        csv_path: Path to the CSV file
        participants: List of participant names to include. If None, includes all with data.
                     Options: ['shervin', 'caitlin', 'ash', 'kelly']

    Returns:
        dict: Nested dictionary in format db[day][participant] = notes_string
              Also includes db[day]['Onyx'] = official_notes
    """
    df = pd.read_csv(csv_path)

    # Default to all participants if not specified
    if participants is None:
        participants = ['shervin', 'caitlin', 'ash', 'kelly']

    # Normalize participant names to lowercase
    participants = [p.lower() for p in participants]

    db = {}

    for _, row in df.iterrows():
        # Use the first 'Day' column (capitalized)
        day = row['Day']

        # Skip rows without valid day numbers (like summary rows)
        try:
            day = int(day)
        except (ValueError, TypeError):
            continue
        db[day] = {}

        # Load participant notes
        for participant in participants:
            notes_col = f'{participant}_notes'
            if notes_col in df.columns:
                notes = row[notes_col]
                # Only add if notes exist and aren't empty
                if pd.notna(notes) and str(notes).strip():
                    # Capitalize first letter for consistency with old format
                    db[day][participant.capitalize()] = str(notes)

        # Load official Onyx notes
        if pd.notna(row['onyx_notes']) and str(row['onyx_notes']).strip():
            db[day]['Onyx'] = str(row['onyx_notes'])

    return db


def get_participant_ratings(csv_path='./data/onyx_advent_2025.csv'):
    """
    Load participant ratings from CSV file.

    Returns:
        dict: Nested dictionary in format ratings[day][participant] = rating
    """
    df = pd.read_csv(csv_path)

    ratings = {}

    for _, row in df.iterrows():
        day = row['Day']

        try:
            day = int(day)
        except (ValueError, TypeError):
            continue
        ratings[day] = {}

        # Load ratings for each participant
        for participant in ['shervin', 'caitlin', 'ash', 'kelly']:
            rating_col = f'{participant}_rating'
            if rating_col in df.columns:
                rating = row[rating_col]
                if pd.notna(rating):
                    ratings[day][participant.capitalize()] = float(rating)

    return ratings


if __name__ == '__main__':
    # Test loading all participants
    print("Loading all participants:")
    db_all = load_notes_from_csv()
    for day in sorted(db_all.keys())[:3]:  # Show first 3 days
        print(f"\nDay {day}:")
        for person, notes in db_all[day].items():
            print(f"  {person}: {notes[:60]}...")

    print("\n" + "="*80)

    # Test loading only Shervin and Caitlin
    print("\nLoading only Shervin and Caitlin:")
    db_two = load_notes_from_csv(participants=['shervin', 'caitlin'])
    for day in sorted(db_two.keys())[:3]:
        print(f"\nDay {day}:")
        for person, notes in db_two[day].items():
            print(f"  {person}: {notes[:60]}...")

    print("\n" + "="*80)

    # Test loading ratings
    print("\nLoading ratings:")
    ratings = get_participant_ratings()
    for day in sorted(ratings.keys())[:3]:
        print(f"\nDay {day}: {ratings[day]}")
