# Nextcloud Talk Multi-Bot Integration

**Status:** ✅ Implemented (2026-01-31)

## Overview

Multiple ask_llm bot personalities can have dedicated Nextcloud Talk rooms. A single webhook endpoint routes messages to the appropriate bot based on conversation token. Provisioning is handled by a separate service at `http://ubuntu.home:8790`.

## Architecture

```
Nextcloud Talk Room (token: abc123)
    ↓ webhook
http://echo.home:8642/webhook/nextcloud
    ↓ routes by conversation_token
ask_llm bot: nova (uses Nova personality)

Nextcloud Talk Room (token: def456)
    ↓ webhook
http://echo.home:8642/webhook/nextcloud
    ↓ routes by conversation_token
ask_llm bot: monika (uses Monika personality)
```

## File Locations

```
src/ask_llm/integrations/nextcloud/
├── __init__.py
├── config.py           # NextcloudBot dataclass, loads from bots.yaml
├── manager.py          # NextcloudBotManager singleton
├── webhook.py          # Webhook handler (routes by conversation_token)
├── provisioner.py      # Client for provisioner service
└── cli.py              # Click-based CLI commands

src/ask_llm/bots.yaml   # Bot configs stored here under each bot's nextcloud: section
src/ask_llm/service/api.py  # POST /webhook/nextcloud, POST /admin/nextcloud-talk/provision
```

## Configuration

### Bot Config (bots.yaml)

Each bot can have a `nextcloud:` section:

```yaml
bots:
  nova:
    name: Nova
    description: Full-featured assistant
    system_prompt: |
      You are Nova...
    nextcloud:
      bot_id: 5                        # Nextcloud bot ID
      secret: "abc123..."              # Bot secret for signing
      conversation_token: "abc123xyz"  # Room token
      enabled: true
```

### Environment Variables

Add to `~/.config/ask-llm/.env`:

```bash
# Provisioner service
ASK_LLM_TALK_PROVISIONER_URL=http://ubuntu.home:8790
ASK_LLM_TALK_PROVISIONER_TOKEN=your-token-here

# Legacy single-bot (deprecated, use bots.yaml)
ASK_LLM_NEXTCLOUD_BOT_SECRET=...
ASK_LLM_NEXTCLOUD_URL=https://nextcloud.ferreri.us
```

## CLI Commands

```bash
# Install/update entry point
./install.sh --local .     # For global pipx install
./install.sh --dev         # For local uv venv only

# List configured bots
llm-nextcloud list                    # Global (pipx)
uv run llm-nextcloud list             # Local venv

# Provision a new bot/room (calls provisioner service)
llm-nextcloud provision --bot nova
llm-nextcloud provision --bot monika --room-name "Monika's Room"

# Rename a room
llm-nextcloud rename --bot nova --name "Nova Assistant"

# Remove config
llm-nextcloud remove --bot nova
```

## API Endpoints

### POST /webhook/nextcloud
Receives webhooks from Nextcloud Talk, routes to appropriate bot.

**Request Headers:**
- `X-Nextcloud-Talk-Signature` - HMAC-SHA256 signature
- `X-Nextcloud-Talk-Random` - Random nonce
- `X-Nextcloud-Talk-Backend` - Nextcloud URL

**Request Body (Activity Streams 2.0):**
```json
{
  "type": "Create",
  "actor": {"id": "users/nick", "name": "nick"},
  "object": {
    "content": "{\"message\":\"Hello bot\"}"
  },
  "target": {
    "id": "abc123xyz",  // conversation_token
    "name": "Nova"
  }
}
```

**Response:**
```json
{
  "status": "received",
  "bot": "nova",
  "message": "Hello bot",
  "from": "nick"
}
```

### POST /admin/nextcloud-talk/provision
Provision a new bot/room via provisioner service.

**Request:**
```json
{
  "bot_id": "nova",
  "room_name": "Nova",  // optional
  "bot_name": "Nova",   // optional
  "owner_user_id": "nick"
}
```

**Response:**
```json
{
  "bot_id": "nova",
  "room_token": "abc123xyz",
  "room_url": "https://nextcloud.ferreri.us/call/abc123xyz",
  "nextcloud_bot_id": 5,
  "nextcloud_bot_name": "Nova"
}
```

## Provisioner Service

**Base URL:** `http://ubuntu.home:8790`
**Auth:** `Authorization: Bearer <TALK_PROVISIONER_TOKEN>`

### POST /provision/nextcloud-talk

**Request:**
```json
{
  "roomName": "Nova",
  "botName": "Nova",
  "webhookUrl": "http://echo.home:8642/webhook/nextcloud",
  "ownerUserId": "nick"
}
```

**Response:**
```json
{
  "nextcloudBaseUrl": "https://nextcloud.ferreri.us",
  "roomToken": "abc123xyz",
  "roomUrl": "https://nextcloud.ferreri.us/call/abc123xyz",
  "botId": 5,
  "botName": "Nova",
  "botSecret": "...",
  "webhookUrl": "http://echo.home:8642/webhook/nextcloud"
}
```

### POST /room/rename

**Request:**
```json
{
  "token": "abc123xyz",
  "name": "New Room Name"
}
```

## Message Signing

### Inbound (Nextcloud → ask_llm)
Signature verification: `HMAC-SHA256(random + body, secret)`

### Outbound (ask_llm → Nextcloud)
**IMPORTANT:** Signature is over `random + messageText`, NOT `random + JSON body`.

```python
signature = hmac.new(
    secret.encode('utf-8'),
    (random_string + message_text).encode('utf-8'),  # NOT the full JSON!
    hashlib.sha256
).hexdigest()
```

Endpoint: `POST /ocs/v2.php/apps/spreed/api/v1/bot/{token}/message`

Headers:
- `X-Nextcloud-Talk-Bot-Random: <random>`
- `X-Nextcloud-Talk-Bot-Signature: <signature>`

Body: `{"message": "response text"}`

## Key Implementation Details

1. **Routing:** Webhook handler uses `conversation_token` from payload to look up bot config
2. **Config Storage:** Bot configs stored in `bots.yaml` (not separate file)
3. **Secrets:** Redacted in logs via `ProvisionResult.__repr__`
4. **Isolation:** Each bot has own secret, conversation context, and personality
5. **Provisioning:** Handled by external service (not manual `occ` commands)

## Testing

```bash
# 1. Set provisioner token
echo 'ASK_LLM_TALK_PROVISIONER_TOKEN=your-token' >> ~/.config/ask-llm/.env

# 2. Provision a bot
llm-nextcloud provision --bot nova

# 3. Check it was added
llm-nextcloud list

# 4. Send message in Nextcloud Talk room
# 5. Verify bot responds with correct personality

# 6. Provision second bot
llm-nextcloud provision --bot monika

# 7. Verify both bots work independently in their rooms
```

## Troubleshooting

**Command not found: llm-nextcloud**
```bash
./install.sh --local .   # For global (pipx)
uv run llm-nextcloud     # For local venv
```

**Provisioning fails with 401**
- Check `ASK_LLM_TALK_PROVISIONER_TOKEN` is set
- Verify token is correct

**Bot not responding**
- Check `llm-nextcloud list` shows the bot
- Verify conversation_token matches room
- Check service logs: `./start.sh logs`

**Wrong bot personality responding**
- Verify conversation_token in bots.yaml matches room
- Check webhook logs for routing decision
