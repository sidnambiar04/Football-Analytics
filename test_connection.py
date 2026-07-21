from database.db import get_connection

conn = get_connection()

if conn:
    print("✅ Test Successful!")
    conn.close()
    print("🔒 Connection Closed")