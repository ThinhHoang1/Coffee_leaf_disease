# auth.py

import sqlite3

# Function to check login credentials
def check_login(user_id, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Query to check if the user exists and the password matches
    c.execute('SELECT * FROM users WHERE user_id = ? AND password = ?', (user_id, password))
    user = c.fetchone()

    conn.close()
    
    return user

# Function to register a new user
def register_user(user_id, name, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Insert a new user into the database
    try:
        c.execute('INSERT INTO users (user_id, name, password) VALUES (?, ?, ?)', (user_id, name, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return False  # User ID already exists
    
    conn.close()
    return True

# Function to create the initial user table (if it doesn't exist)
def create_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create a table for storing user information
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        password TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Call this function to create the table when the script is first run
create_table()
