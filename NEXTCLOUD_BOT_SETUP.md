# Nextcloud Talk Bot Setup Guide

## Step 1: Start the Test Webhook Receiver

```bash
# Install Flask if needed
pip install flask

# Run the test webhook receiver
python test_webhook.py
```

This will start a server on `http://localhost:5000` that logs all incoming webhooks.

## Step 2: Register the Bot with Nextcloud

**On your Nextcloud server**, run the `occ` command:

```bash
# Basic syntax
sudo -u www-data php occ talk:bot:install \
  <name> \
  <shared-secret> \
  <webhook-url>

# Example
sudo -u www-data php occ talk:bot:install \
  "AskLLM Bot" \
  "my-super-secret-key-12345" \
  "http://your-server:5000/webhook/nextcloud"
```

**Important:**
- The webhook URL must be reachable from your Nextcloud server
- If testing locally, you might need ngrok or similar to expose localhost
- Save the secret! You'll need it in both test scripts

### Using ngrok (if Nextcloud can't reach your local machine)

```bash
# Install ngrok: https://ngrok.com/
ngrok http 5000

# Use the ngrok URL (e.g., https://abc123.ngrok.io/webhook/nextcloud)
```

## Step 3: Update Test Script Secrets

Edit `test_webhook.py`:
```python
SHARED_SECRET = "my-super-secret-key-12345"  # Must match occ command
```

Edit `test_nextcloud_send.py`:
```python
NEXTCLOUD_URL = "https://nextcloud.example.com"
BOT_SECRET = "my-super-secret-key-12345"
CONVERSATION_TOKEN = "abc123xyz"  # Get from Talk room
```

## Step 4: Add Bot to a Talk Room

In Nextcloud Talk web interface:
1. Open a conversation
2. Click the "..." menu (top right)
3. Select "Conversation settings"
4. Go to "Bots" section
5. Select your bot from the list
6. Click "Add"

## Step 5: Test Receiving Messages

1. Send a message in the Talk room
2. Check your `test_webhook.py` console output
3. You should see the webhook payload with signature verification

## Step 6: Test Sending Messages

Get the conversation token:
1. In Talk, go to conversation settings
2. Look at the URL: `.../call/{token}`
3. Copy that token

Run the send test:
```bash
python test_nextcloud_send.py "Hello from bot!"
```

## Useful occ Commands

```bash
# List all bots
sudo -u www-data php occ talk:bot:list

# List bots in a specific conversation
sudo -u www-data php occ talk:bot:state <conversation-token>

# Remove a bot
sudo -u www-data php occ talk:bot:uninstall <bot-id>
```

## Debugging

If webhooks aren't arriving:
1. Check Nextcloud logs: `tail -f /var/log/nextcloud/nextcloud.log`
2. Verify webhook URL is accessible: `curl http://your-server:5000/health`
3. Check firewall rules
4. Verify bot is added to the conversation

If signature validation fails:
1. Double-check the secret matches exactly
2. Verify you're using UTF-8 encoding
3. Check the signature calculation matches: `HMAC-SHA256(random + body, secret)`

## Expected Webhook Payload

When someone sends "Hello bot!" you should see:

```json
{
  "type": "Create",
  "actor": {
    "type": "Person",
    "id": "users/alice",
    "name": "Alice"
  },
  "object": {
    "type": "Note",
    "id": "message-id",
    "name": "Hello bot!",
    "mediaType": "text/markdown"
  },
  "target": {
    "type": "Collection",
    "id": "conversation-token",
    "name": "Room Name"
  }
}
```

## Next Steps

Once you can receive and send messages manually:
1. Integrate with ask_llm service
2. Add proper error handling
3. Support message threading/replies
4. Handle different message types (reactions, etc.)
