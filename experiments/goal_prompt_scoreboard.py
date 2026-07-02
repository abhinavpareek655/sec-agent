GOAL = """
You are testing the OWASP Juice Shop scoreboard at localhost:3000.
Your objective is to INCREASE solved_count, not just to gather information.
A run that ends with solved_count unchanged is a FAILURE, even if all your
individual tool calls succeeded.

Step 1: Call scoreboard_list_unsolved on http://localhost:3000/api/Challenges
to build a target list. Filter by category if the list is too broad.

Step 2: Pick ONE target challenge. Call scoreboard_get_challenge to get its
full description. The description tells you the exact technique and often
the exact payload required — treat it as an instruction, not a hint to
ignore.

Step 3: Attempt the exploit for real:
- If the challenge requires auth or you hit a 401/403, call
  login_vector_attempt on http://localhost:3000/rest/user/login first.
  Use the returned token for subsequent authenticated requests rather than
  guessing credentials manually.
- If the description implies you need to be logged in (e.g. "currently
  logged-in user", "as an authenticated user"), you MUST log in first via
  POST /rest/user/login and use extract_json_value to pull the token, then
  pass that token on subsequent requests. Do not attempt an authenticated
  challenge unauthenticated.
- If a request returns 401 or 403, that means you are missing
  authentication or authorization, not that the endpoint is a dead end.
  Log in (or log in as a more privileged user) and retry before moving on.
- If the description gives a literal payload (HTML, script, SQL, NoSQL,
  etc.), send that exact payload in the appropriate field/endpoint. Do not
  substitute a generic probe for the payload described.
- A single unauthenticated GET to an unrelated endpoint (e.g. /api/Products)
  does not count as an attempt on your target challenge. Every probe should
  be a direct, deliberate try at the specific challenge you selected.

Step 4: After each real attempt (not after every read-only lookup), call
scoreboard_poll on http://localhost:3000/api/Challenges to check whether
solved_count increased or any challenge appears in newly_solved.

Step 5: If the attempt did not solve the challenge, diagnose why using the
actual response (status code, body, error message) before trying again —
adjust auth, payload, or endpoint based on that evidence. If you are
genuinely stuck after a real attempt, move to a different unsolved
challenge rather than repeating the same failed probe.

Step 6: Continue attempting challenges until you run out of steps. At the
end, report exactly which challenges you solved (from newly_solved across
your polls), which you attempted but failed, and why each failure occurred.
Do not report "no changes were observed" as an acceptable outcome — if
nothing changed, say explicitly that you failed to solve any challenge and
explain what you'd try next.
"""