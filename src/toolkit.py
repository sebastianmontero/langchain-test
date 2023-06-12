from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.tools.sql_database.tool import InfoSQLDatabaseTool
from langchain.tools import BaseTool
from typing import List


class ConservativeSQLDatabaseToolkit(SQLDatabaseToolkit):
    
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        info_sql_database_tool_description = (
            "Input to this tool is the name of a table, output is the "
            "schema and sample rows for the table. "
            "Be sure that the table exists by calling list_tables_sql_db "
            "first! Example Input: 'table1'"
        )
        tools = super().get_tools()
        for tool in tools:
            if isinstance(tool, InfoSQLDatabaseTool):
                tool.description = info_sql_database_tool_description
        return tools


