from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Conversation, Message, utc_now


class ConversationNotFound(Exception):
    pass


class ConversationRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    def list_conversations(self, limit: int, offset: int) -> list[Conversation]:
        query = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with self.sessions() as session:
            return list(session.scalars(query))

    def list_messages(
        self, conversation_id: UUID, limit: int, offset: int = 0
    ) -> list[Message]:
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with self.sessions() as session:
            if session.get(Conversation, conversation_id) is None:
                raise ConversationNotFound
            messages = list(session.scalars(query))
            messages.reverse()
            return messages

    def save_turn(self, conversation_id: UUID | None, user: str, answer: str) -> UUID:
        with self.sessions.begin() as session:
            now = utc_now()
            if conversation_id is None:
                conversation = Conversation(created_at=now, updated_at=now)
                session.add(conversation)
                session.flush()
                conversation_id = conversation.id
            else:
                result = session.execute(
                    update(Conversation)
                    .where(Conversation.id == conversation_id)
                    .values(updated_at=now)
                )
                if result.rowcount == 0:
                    raise ConversationNotFound

            session.add_all([
                Message(conversation_id=conversation_id, role="user", content=user),
                Message(conversation_id=conversation_id, role="assistant", content=answer),
            ])
        return conversation_id

    def delete_conversation(self, conversation_id: UUID) -> None:
        with self.sessions.begin() as session:
            result = session.execute(
                delete(Conversation).where(Conversation.id == conversation_id)
            )
            if result.rowcount == 0:
                raise ConversationNotFound
