import os
from logging import Logger
from pathlib import Path

from langrila import Agent, Prompt, SystemPrompt
from langrila.core.response import TextResponse
from langrila.core.typing import (
    ClientMessage,
    ClientMessageContent,
    ClientSystemMessage,
    ClientTool,
)

from ..base_extractor import BaseExtractor
from ..io_data import Inputs, Keyphrase, Outputs
from .data import ResponseSchema


PROMPT_DIRPATH = Path(os.path.abspath(__file__)).parent / "prompt_texts"


class GenerationBasedExtractor(BaseExtractor):
    """
    A keyphrase extractor using a generative agent to extract and rank keyphrases.

    This class utilizes an AI agent to extract keyphrases based on prompts and system
    instructions. It supports customization of system prompts and handles multi-document
    extraction.

    Attributes:
        agent (Agent): The AI agent used for generating keyphrases.
        system_prompt (SystemPrompt): The system prompt containing instructions for
                                      the agent.
    """

    def __init__(
        self,
        agent: Agent[
            ClientMessage, ClientSystemMessage, ClientMessageContent, ClientTool
        ],
        system_prompt: SystemPrompt | str | Path | None = None,
        max_characters: int | None = None,
        flat_output: bool = True,
        use_order: bool = False,
        rrf_k: int = 60,
        logger: Logger | None = None,
    ):
        """
        Initializes the GenerationBasedExtractor with an agent and optional system
        prompts.

        Args:
            agent (Agent): The AI agent for keyphrase generation.
            system_prompt (SystemPrompt | str | Path | None): The system prompt,
                provided as a SystemPrompt object, string, or file path.
            max_characters (int | None): Maximum characters per chunk of text.
            flat_output (bool): Whether to flatten the output structure.
            use_order (bool): Whether to consider order during ranking.
            rrf_k (int): Parameter for Reciprocal Rank Fusion (RRF) scoring.
            logger (Logger | None): Logger instance or None for no logging.
        """
        super().__init__(set(), max_characters, flat_output, use_order, rrf_k, logger)
        self.agent = agent

        if isinstance(system_prompt, SystemPrompt):
            self.system_prompt = system_prompt
        elif isinstance(system_prompt, str):
            self.system_prompt = SystemPrompt(
                role="system",
                contents=system_prompt,
            )
        elif isinstance(system_prompt, Path):
            self.system_prompt = self._load_system_prompt(prompt_filepath=system_prompt)
        else:
            self.system_prompt = self._load_system_prompt()

        if self.logger:
            self.logger.info(f"System prompt: {self.system_prompt}")

    def _load_system_prompt(
        self,
        prompt_filepath: Path = PROMPT_DIRPATH / "japanese_keyphrase_extraction.txt",
    ) -> SystemPrompt:
        """
        Loads a system prompt from the specified file.

        Args:
            prompt_filepath (Path): The file path to the system prompt text.

        Returns:
            SystemPrompt: The loaded system prompt.
        """
        return SystemPrompt(
            role="system",
            contents=prompt_filepath.read_text(),
        )

    def _make_user_prompt(self, text: str, top_n_phrases: int) -> Prompt:
        """
        Creates a user prompt for the AI agent.

        Args:
            text (str): The text for which keyphrases are extracted.
            top_n_phrases (int): The number of keyphrases to extract.

        Returns:
            Prompt: The generated user prompt.
        """
        return Prompt(
            role="user",
            contents=f"N={top_n_phrases}\n文章:\n{text}",
        )

    def _extract(self, text: str, top_n_phrases: int) -> list[Keyphrase]:
        """
        Extracts keyphrases from the text using the AI agent.

        Args:
            text (str): The input text for keyphrase extraction.
            top_n_phrases (int): The number of keyphrases to extract.

        Returns:
            list[Keyphrase]: A sorted list of extracted keyphrases.
        """
        user_prompt = self._make_user_prompt(text=text, top_n_phrases=top_n_phrases)
        if self.logger:
            self.logger.info(f"User prompt: {user_prompt}")

        try:
            response = self.agent.generate_text(
                prompt=user_prompt, system_instruction=self.system_prompt
            )
            if self.logger:
                self.logger.info(f"Response: {response}")
        except Exception as e:
            print(e)
            return []

        if isinstance(response.contents[0], TextResponse):
            _keyphrases = ResponseSchema.model_validate_json(response.contents[0].text)
            return sorted(_keyphrases.keyphrases, key=lambda x: x.score, reverse=True)
        else:
            raise ValueError(
                "Response contents[0] is not of type TextResponse. Actual type: "
                f"{type(response.contents[0])}"
            )

    def get_keyphrase(
        self, input_text: str | list[str] | Inputs, top_n_phrases: int = 10
    ) -> Outputs:
        """
        Extracts keyphrases from the input text using the generative agent.

        Args:
            input_text (str | list[str] | Inputs): The input text(s) or preprocessed data.
            top_n_phrases (int): The number of keyphrases to extract.

        Returns:
            Outputs: Extracted keyphrases with their corresponding scores.
        """
        verify_input: Inputs = self._verify_input(input_text=input_text)
        docs: list[str] = verify_input.docs
        outputs: Outputs = Outputs(
            keyphrases=[
                self._extract(text=doc, top_n_phrases=top_n_phrases) for doc in docs
            ]
        )
        if self.logger:
            self.logger.debug(f"Outputs: {outputs.model_dump_json(indent=4)}")

        if self.flat_output and len(outputs.keyphrases) > 1:
            outputs = self._flatten_outputs(
                keyphrases_list=outputs.keyphrases,
                use_order=self.use_order,
                descending=True,
                rrf_k=self.rrf_k,
            )
        return outputs
