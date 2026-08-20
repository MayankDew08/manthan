import os
from urllib.parse import urlparse

from neo4j import GraphDatabase

_URL_NOISE_SEGMENTS = {
    "status", "statuses", "blob", "tree", "raw", "tag", "tags",
    "pull", "issues", "releases", "topics", "posts", "products",
    "watch", "starred", "users", "u", "profile", "category",
    "article", "articles", "blog", "home", "index", "main",
    "default", "overview", "readme",
}


def _url_to_title(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = host.removeprefix("www.").split(":")[0]
    path = urlparse(url).path.strip("/")
    segments = [
        seg for seg in path.split("/")
        if seg and seg not in _URL_NOISE_SEGMENTS and not seg.isdigit()
    ]
    keep = segments[-2:]
    return host + ("/" + "/".join(keep) if keep else "")


class KnowledgeStore:
    def __init__(self, uri=None, user=None, password=None):
        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                  password or os.environ.get("NEO4J_PASSWORD", "")),
        )

    def close(self):
        self.driver.close()

    def add_message(self, msg: dict):
        with self.driver.session() as session:
            session.execute_write(self._add_message_tx, msg)

    @staticmethod
    def _add_message_tx(tx, msg):
        sender = msg.get("sender") or ""
        sent_at = msg.get("sent_at") or msg.get("datetime_iso") or ""
        text = msg.get("original_text") or ""
        tx.run(
            """
            MERGE (p:Person {name: $sender})
            MERGE (m:Message {sent_at: $sent_at, sender: $sender, text: $text})
            ON CREATE SET m.quality = $quality, m.link_intent = $link_intent,
                m.entities = $entities, m.topics = $topics, m.trusted = $trusted
            ON MATCH SET m.quality = $quality, m.link_intent = $link_intent,
                m.entities = $entities, m.topics = $topics, m.trusted = $trusted
            MERGE (p)-[:SENT]->(m)
            WITH m
            FOREACH (name IN $entities |
                MERGE (e:Entity {name: name})
                MERGE (m)-[:MENTIONS]->(e))
            FOREACH (name IN $topics |
                MERGE (t:Topic {name: name})
                MERGE (m)-[:ABOUT]->(t))
            WITH m
            UNWIND $links AS link
            MERGE (l:Link {url: link.url})
            ON CREATE SET l.title = link.title
            MERGE (m)-[:CONTAINS]->(l)
            """,
            sender=sender,
            sent_at=sent_at,
            text=text,
            quality=msg.get("quality"),
            link_intent=msg.get("link_intent") or "",
            entities=msg.get("entities") or [],
            topics=msg.get("topics") or [],
            trusted=bool(msg.get("trusted")),
            links=[{"url": url, "title": _url_to_title(url)}
                   for url in msg.get("links") or []],
        )

    def add_link(self, link: dict, status: str = "scraped"):
        with self.driver.session() as session:
            session.execute_write(self._add_link_tx, link, status)

    @staticmethod
    def _add_link_tx(tx, link, status):
        url = link.get("url") or ""
        ctx = link.get("message_context") or {}
        tx.run(
            """
            MERGE (l:Link {url: $url})
            SET l.final_url = $final_url,
                l.title = $title,
                l.source = $source,
                l.raw_text = $raw_text,
                l.summary = $summary,
                l.what_it_is = $what_it_is,
                l.problem_solved = $problem_solved,
                l.how_useful = $how_useful,
                l.link_intent = $link_intent,
                l.scraped_at = $scraped_at,
                l.sent_at = $sent_at,
                l.status = $status
            WITH l
            FOREACH (name IN $entities |
                MERGE (e:Entity {name: name})
                MERGE (l)-[:MENTIONS]->(e))
            FOREACH (name IN $topics |
                MERGE (t:Topic {name: name})
                MERGE (l)-[:ABOUT]->(t))
            WITH l
            MATCH (m:Message {sender: $sender, sent_at: $msg_sent_at, text: $text})
            MERGE (m)-[:CONTAINS]->(l)
            """,
            url=url,
            final_url=link.get("final_url") or url,
            title=link.get("title") or "",
            source=link.get("source") or "",
            raw_text=link.get("raw_text") or "",
            summary=link.get("summary") or "",
            what_it_is=link.get("what_it_is") or "",
            problem_solved=link.get("problem_solved") or "",
            how_useful=link.get("how_useful") or "",
            link_intent=link.get("link_intent") or "",
            scraped_at=link.get("scraped_at") or "",
            sent_at=link.get("sent_at") or "",
            entities=link.get("entities") or [],
            topics=link.get("topics") or [],
            sender=ctx.get("sender") or "",
            msg_sent_at=ctx.get("sent_at") or "",
            text=ctx.get("original_text") or "",
            status=status,
        )

    def add_pending_link(self, record: dict):
        with self.driver.session() as session:
            session.execute_write(self._add_pending_link_tx, record)

    @staticmethod
    def _add_pending_link_tx(tx, record):
        url = record.get("url") or ""
        tx.run(
            """
            MERGE (l:Link {url: $url})
            SET l.status = 'pending_paste',
                l.block_reason = $block_reason,
                l.partial_text = $partial_text,
                l.link_intent = $link_intent,
                l.sent_at = $sent_at,
                l.title = coalesce(l.title, $title)
            """,
            url=url,
            block_reason=record.get("block_reason") or "",
            partial_text=record.get("partial_text") or "",
            link_intent=((record.get("message_context") or {}).get("link_intent")) or "",
            sent_at=record.get("sent_at") or "",
            title=_url_to_title(url),
        )

    def pending_links(self) -> list:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (l:Link)
                WHERE l.status = 'pending_paste'
                RETURN l.url AS url, l.title AS title,
                       l.block_reason AS block_reason,
                       l.partial_text AS partial_text,
                       l.link_intent AS link_intent,
                       l.sent_at AS sent_at
                ORDER BY l.sent_at
                """
            ).data()
        return rows