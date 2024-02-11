import json

# List of column names
column_names = [
    "MPAA Rating_G", "MPAA Rating_GP", "MPAA Rating_M/PG", "MPAA Rating_NC-17", "MPAA Rating_Not Rated",
    "MPAA Rating_Open", "MPAA Rating_PG", "MPAA Rating_PG-13", "MPAA Rating_R",
    "Keywords_1930s", "Keywords_1940s", "Keywords_1950s", "Keywords_1960s", "Keywords_1970s", "Keywords_1980s", "Keywords_1990s", "Keywords_2000s", "Keywords_3-D",
    "Keywords_3-D - Post-production Conversion", "Keywords_3-D - Shot in 3-D", "Keywords_Accidental Death", "Keywords_Action Adventure", "Keywords_Action Comedy",
    "Keywords_Action Horror", "Keywords_Action Thriller", "Keywords_Addiction", "Keywords_Adopted Family", "Keywords_African Americans", "Keywords_Alien Invasion",
    "Keywords_Animal Lead", "Keywords_Animals Gone Bad", "Keywords_Artists", "Keywords_Autobiographical", "Keywords_Autumn Years", "Keywords_Betrayal", "Keywords_Bigotry",
    "Keywords_Biographical Drama", "Keywords_Biography", "Keywords_Black Comedy", "Keywords_Blackmail", "Keywords_Breaking the Fourth Wall", "Keywords_Buddy Comedy",
    "Keywords_Buddy Cop", "Keywords_Bullies", "Keywords_C.I.A.", "Keywords_Cancer", "Keywords_Car Accident", "Keywords_Child Abuse", "Keywords_Christmas", "Keywords_College",
    "Keywords_Comedy Drama", "Keywords_Coming of Age", "Keywords_Confidence Men", "Keywords_Conspiracy Theory", "Keywords_Corporate Malfeasance", "Keywords_Corrupt Cops",
    "Keywords_Crime", "Keywords_Crime Comedy", "Keywords_Crime Drama", "Keywords_Crime Thriller", "Keywords_Cross-Class Romance", "Keywords_Cross-Dressing", "Keywords_Cult Movie",
    "Keywords_Culture Clash", "Keywords_Dancing", "Keywords_Death of a Sibling", "Keywords_Death of a Son or Daughter", "Keywords_Death of a Spouse or Fiancée / Fiancé",
    "Keywords_Delayed Adulthood", "Keywords_Delayed Sequel", "Keywords_Demons", "Keywords_Depression", "Keywords_Development Hell", "Keywords_Directing Yourself",
    "Keywords_Disaster", "Keywords_Doctors", "Keywords_Domestic Abuse", "Keywords_Dream Sequence", "Keywords_Dysfunctional Family", "Keywords_Dystopia", "Keywords_End of the World",
    "Keywords_Ensemble", "Keywords_Epilogue", "Keywords_Erotic Thriller", "Keywords_FBI", "Keywords_Faith-Based Film", "Keywords_Faked Death", "Keywords_False Identity",
    "Keywords_Family Affair", "Keywords_Family Comedy", "Keywords_Family Drama", "Keywords_Family Movie", "Keywords_Famously Bad", "Keywords_Farcical / Slapstick Comedy",
    "Keywords_Father’s Footsteps", "Keywords_Faulty Memory", "Keywords_Film Noir", "Keywords_Fired", "Keywords_First Love", "Keywords_Food", "Keywords_Football",
    "Keywords_Foreign Language", "Keywords_Foreign-Language Remake", "Keywords_Framed", "Keywords_Fugitive / On the Run", "Keywords_Gambling", "Keywords_Gangs",
    "Keywords_Good vs. Evil", "Keywords_Government Corruption", "Keywords_Gratuitous Cameos", "Keywords_Hallucinations", "Keywords_Haunting", "Keywords_Heist", "Keywords_High School",
    "Keywords_Historical Drama", "Keywords_Hitmen", "Keywords_Hood Film", "Keywords_Horror Comedy", "Keywords_Hostage", "Keywords_IMAX: DMR", "Keywords_Immigration",
    "Keywords_In a Plane", "Keywords_Infidelity", "Keywords_Inheritance", "Keywords_Inspired by a True Story", "Keywords_Internet", "Keywords_Intertitle", "Keywords_Inventor",
    "Keywords_Investigative Journalist", "Keywords_Jewish", "Keywords_Kidnap", "Keywords_LGBTQ+", "Keywords_Lawyers", "Keywords_Life Drama", "Keywords_Life in a Small Town",
    "Keywords_Life on the Outside", "Keywords_Love Triangle", "Keywords_Mafia", "Keywords_Martial Arts", "Keywords_Marvel Comics", "Keywords_Medical and Hospitals",
    "Keywords_Mental Illness", "Keywords_Mid-Life Crisis", "Keywords_Missing Person", "Keywords_Mistaken Identity", "Keywords_Money Troubles", "Keywords_Monster",
    "Keywords_Motion Capture Performance", "Keywords_Movie Business", "Keywords_Music Industry", "Keywords_Musicians", "Keywords_Narcotics", "Keywords_Near Future",
    "Keywords_New Guy/Girl in School", "Keywords_News", "Keywords_No Honor Among Thieves", "Keywords_Non-Chronological", "Keywords_Novel or Other Work Adapted by Author",
    "Keywords_Organized Crime", "Keywords_Orphan", "Keywords_Oscars Best Picture Winner", "Keywords_Other_Keywords", "Keywords_Performing Arts", "Keywords_Police Detective",
    "Keywords_Political", "Keywords_Possessed", "Keywords_Post Apocalypse", "Keywords_Posthumous Release", "Keywords_Pregnant Women", "Keywords_Prison",
    "Keywords_Prison Break", "Keywords_Private Investigator", "Keywords_Professional Rivalry", "Keywords_Prologue", "Keywords_Prostitution", "Keywords_Psychological Horror",
    "Keywords_Psychological Thriller", "Keywords_Relationship Advice", "Keywords_Relationships Gone Wrong", "Keywords_Religious", "Keywords_Remake", "Keywords_Rescue",
    "Keywords_Returning Soldiers", "Keywords_Revenge", "Keywords_Road Trip", "Keywords_Robot", "Keywords_Rock 'n' Roll", "Keywords_Romance", "Keywords_Romantic Comedy",
    "Keywords_Romantic Drama", "Keywords_Royalty", "Keywords_Same Actor, Multiple Roles", "Keywords_Same Role, Multiple Actors", "Keywords_Satire", "Keywords_Satirical Comedy",
    "Keywords_Scene in End Credits", "Keywords_Screenplay Written By the Star", "Keywords_Secret Agent", "Keywords_Segments", "Keywords_Serial Killer", "Keywords_Set in Los Angeles",
    "Keywords_Set in New York", "Keywords_Set in New York City", "Keywords_Sex Crimes", "Keywords_Sibling Rivalry", "Keywords_Singers", "Keywords_Single Parent",
    "Keywords_Slasher Horror", "Keywords_Spoof", "Keywords_Sports Drama", "Keywords_Suicide", "Keywords_Supernatural", "Keywords_Supernatural Horror", "Keywords_Surprise Twist",
    "Keywords_TV Industry", "Keywords_Talking Animals", "Keywords_Terminal Illness", "Keywords_Terrorism", "Keywords_Time Travel", "Keywords_Twins", "Keywords_Undercover",
    "Keywords_Underdog", "Keywords_Unexpected Families", "Keywords_Unnamed Character", "Keywords_Vampire", "Keywords_Vietnam War", "Keywords_Visual Effects",
    "Keywords_Voiceover/Narration", "Keywords_War", "Keywords_War Drama", "Keywords_Wedding Day", "Keywords_White Collar Crime", "Keywords_Widow/Widower", "Keywords_World War II",
    "Keywords_Writing and Writers", "Keywords_Young Child Dealing with the Death of a Parent", "Keywords_Zombies", "Source_Based on Ballet", "Source_Based on Comic/Graphic Novel",
    "Source_Based on Factual Book/Article", "Source_Based on Fiction Book/Short Story", "Source_Based on Folk Tale/Legend/Fairytale", "Source_Based on Game", "Source_Based on Movie",
    "Source_Based on Musical Group", "Source_Based on Musical or Opera", "Source_Based on Play", "Source_Based on Poem", "Source_Based on Real Life Events", "Source_Based on Religious Text",
    "Source_Based on Short Film", "Source_Based on Song", "Source_Based on TV", "Source_Based on Theme Park Ride", "Source_Based on Toy", "Source_Based on Web Series", "Source_Compilation",
    "Source_Original Screenplay", "Source_Remake", "Source_Spin-Off", "Production method_Animation/Live Action", "Production method_Digital Animation", "Production method_Hand Animation",
    "Production method_Live Action", "Production method_Multiple Production Methods", "Production method_Rotoscoping", "Production method_Stop-Motion Animation",
    "Creative type_Contemporary Fiction", "Creative type_Dramatization", "Creative type_Factual", "Creative type_Fantasy", "Creative type_Historical Fiction", "Creative type_Kids Fiction",
    "Creative type_Multiple Creative Types", "Creative type_Science Fiction", "Creative type_Super Hero",
    "Countries_Australia", "Countries_Canada", "Countries_China", "Countries_France", "Countries_Germany", "Countries_India", "Countries_Italy", "Countries_Japan", "Countries_Other_Countries",
    "Countries_Spain", "Countries_United Kingdom", "Countries_United States"
]

# Store column_names in a JSON file
json_filename = 'column_names.json'
with open(json_filename, 'w') as json_file:
    json.dump(column_names, json_file)

print(f"Column names stored in '{json_filename}'")
