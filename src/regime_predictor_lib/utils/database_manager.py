import logging
from pathlib import Path

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for database: {self.db_path.parent}")

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"DatabaseManager initialized for SQLite database at: {self.db_path}")

    def get_session(self):
        return self._SessionLocal()

    def _build_upsert_statement(
        self,
        table_name: str,
        df_columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> str:
        cols_str = ", ".join(df_columns)
        placeholders_str = ", ".join([f":{col}" for col in df_columns])
        conflict_cols_str = ", ".join(conflict_columns)

        if update_columns is None:
            update_columns = [col for col in df_columns if col not in conflict_columns]

        if not update_columns:
            set_clause = "NOTHING"
            statement = (
                f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders_str}) "
                f"ON CONFLICT({conflict_cols_str}) DO {set_clause}"
            )

        else:
            set_clause_parts = [f"{col} = excluded.{col}" for col in update_columns]
            set_clause = ", ".join(set_clause_parts)
            statement = (
                f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders_str}) "
                f"ON CONFLICT({conflict_cols_str}) DO UPDATE SET {set_clause}"
            )
        return statement

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
        chunksize: int = 1000,
    ):
        if df.empty:
            logger.info(f"DataFrame for table '{table_name}' is empty. Nothing to upsert.")
            return

        statement = self._build_upsert_statement(
            table_name,
            df.columns.tolist(),
            conflict_columns,
            update_columns,
        )
        logger.debug(f"Upsert statement for table {table_name}: {statement}")

        num_chunks = (len(df) - 1) // chunksize + 1
        for i in range(num_chunks):
            chunk_df = df.iloc[i * chunksize : (i + 1) * chunksize]
            records = chunk_df.to_dict(orient="records")

            session = self.get_session()
            try:
                session.execute(text(statement), records)
                session.commit()
                logger.info(
                    f"Successfully upserted chunk {i + 1}/{num_chunks} "
                    f"({len(records)} rows) into {table_name}."
                )
            except Exception as e:
                session.rollback()
                logger.error(
                    f"Error upserting data into {table_name} (chunk {i + 1}): {e}",
                    exc_info=True,
                )
                raise
            finally:
                session.close()

    def table_exists(self, table_name: str) -> bool:
        inspector = sqlalchemy.inspect(self.engine)
        return inspector.has_table(table_name)

    def create_tables_from_schema_file(self, schema_file_path: Path | str):
        schema_file = Path(schema_file_path)
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        with open(schema_file, "r") as f:
            schema_sql = f.read()

        statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]

        session = self.get_session()
        try:
            for stmt in statements:
                if stmt:
                    if stmt.upper().startswith("CREATE TABLE"):
                        parts = stmt.split()
                        try:
                            table_name_idx = parts.index("TABLE") + 1
                            if (
                                parts[table_name_idx].upper() == "IF"
                                and parts[table_name_idx + 1].upper() == "NOT"
                                and parts[table_name_idx + 2].upper() == "EXISTS"
                            ):
                                table_name_idx += 3

                            table_name = parts[table_name_idx]

                            table_name = table_name.split("(")[0].replace('"', "").replace("`", "")

                            if not self.table_exists(table_name):
                                logger.info(f"Table '{table_name}' does not exist. Creating...")
                                session.execute(text(stmt))
                                logger.info(f"Executed: {stmt}")
                            else:
                                logger.info(
                                    f"Table '{table_name}' already exists. Skipping creation."
                                )
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Could not parse table name from statement: {stmt}. "
                                f"Error: {e}. Executing anyway."
                            )
                            session.execute(text(stmt))
                    else:
                        session.execute(text(stmt))
                        logger.info(f"Executed: {stmt}")
            session.commit()
            logger.info("Schema loaded/verified successfully.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error executing schema file {schema_file}: {e}", exc_info=True)
            raise
        finally:
            session.close()
