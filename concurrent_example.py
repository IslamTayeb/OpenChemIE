"""
Example: Concurrent version of extract_reactions_from_pdf

This shows the minimal changes needed for threading-based concurrency.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# OPTION 1: Threading (minimal changes, good for CUDA)
def extract_reactions_from_pdf_concurrent_threading(self, pdf, num_pages=None, batch_size=16):
    """
    Concurrent version using threading.
    Best for CUDA operations (releases GIL during GPU inference).
    """
    # Phase 1: Extract figures (needed for later steps)
    figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
    images = [figure['figure']['image'] for figure in figures]

    # Phase 2: Launch independent operations in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit independent tasks
        future_reactions = executor.submit(
            self.extract_reactions_from_figures,
            images, batch_size=batch_size, molscribe=True, ocr=True
        )
        future_text = executor.submit(
            self.extract_reactions_from_text_in_pdf,
            pdf, num_pages=num_pages
        )
        future_coref = executor.submit(
            self.extract_molecule_corefs_from_figures_in_pdf,
            pdf, num_pages=num_pages
        )

        # Wait for all to complete
        results = future_reactions.result()
        text_results = future_text.result()
        results_coref = future_coref.result()

    # Phase 3: Sequential post-processing (dependencies)
    table_expanded_results = process_tables(figures, results, self.molscribe, batch_size=batch_size)
    figure_results = replace_rgroups_in_figure(
        figures, table_expanded_results, results_coref,
        self.molscribe, batch_size=batch_size
    )
    table_expanded_results = expand_reactions_with_backout(
        figure_results, results_coref, self.molscribe
    )
    coref_expanded_results = associate_corefs(text_results, results_coref)

    return {
        'figures': table_expanded_results,
        'text': coref_expanded_results,
    }


# OPTION 2: Multiprocessing (more changes, bypasses GIL completely)
def extract_reactions_from_pdf_concurrent_multiprocessing(self, pdf, num_pages=None, batch_size=16):
    """
    Concurrent version using multiprocessing.
    Best for CPU-only operations (bypasses GIL completely).

    NOTE: This requires models to be reloaded in each process,
    or models must be passed/shared between processes.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    # Phase 1: Extract figures (needed for later steps)
    figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
    images = [figure['figure']['image'] for figure in figures]

    # Helper function for worker processes
    def worker_extract_text(pdf_path, num_pages):
        # Each process needs its own model instance
        extractor = ChemRxnExtractor("", None, self.chemrxnextractor.model_dir, self.device.type)
        extractor.set_pdf_file(pdf_path)
        extractor.set_pages(num_pages)
        return extractor.extract_reactions_from_text()

    # Phase 2: Launch independent operations in parallel processes
    with ProcessPoolExecutor(max_workers=min(3, mp.cpu_count())) as executor:
        future_reactions = executor.submit(
            self._extract_reactions_worker,
            images, batch_size, self.device
        )
        future_text = executor.submit(
            worker_extract_text,
            pdf, num_pages
        )
        future_coref = executor.submit(
            self._extract_coref_worker,
            pdf, num_pages, self.device
        )

        results = future_reactions.result()
        text_results = future_text.result()
        results_coref = future_coref.result()

    # Phase 3: Sequential post-processing (same as threading version)
    table_expanded_results = process_tables(figures, results, self.molscribe, batch_size=batch_size)
    figure_results = replace_rgroups_in_figure(
        figures, table_expanded_results, results_coref,
        self.molscribe, batch_size=batch_size
    )
    table_expanded_results = expand_reactions_with_backout(
        figure_results, results_coref, self.molscribe
    )
    coref_expanded_results = associate_corefs(text_results, results_coref)

    return {
        'figures': table_expanded_results,
        'text': coref_expanded_results,
    }


# OPTION 3: Async/await (most complex, best for I/O-bound)
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def extract_reactions_from_pdf_concurrent_async(self, pdf, num_pages=None, batch_size=16):
    """
    Concurrent version using asyncio.
    Best for I/O-bound operations and handling multiple PDFs.
    """
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=3)

    # Phase 1: Extract figures
    figures = await loop.run_in_executor(
        executor,
        self.extract_figures_from_pdf,
        pdf, num_pages, True
    )
    images = [figure['figure']['image'] for figure in figures]

    # Phase 2: Launch independent operations
    tasks = [
        loop.run_in_executor(
            executor,
            self.extract_reactions_from_figures,
            images, batch_size, True, True
        ),
        loop.run_in_executor(
            executor,
            self.extract_reactions_from_text_in_pdf,
            pdf, num_pages
        ),
        loop.run_in_executor(
            executor,
            self.extract_molecule_corefs_from_figures_in_pdf,
            pdf, batch_size, num_pages, True, True
        ),
    ]

    results, text_results, results_coref = await asyncio.gather(*tasks)

    # Phase 3: Sequential post-processing
    table_expanded_results = process_tables(figures, results, self.molscribe, batch_size=batch_size)
    figure_results = replace_rgroups_in_figure(
        figures, table_expanded_results, results_coref,
        self.molscribe, batch_size=batch_size
    )
    table_expanded_results = expand_reactions_with_backout(
        figure_results, results_coref, self.molscribe
    )
    coref_expanded_results = associate_corefs(text_results, results_coref)

    return {
        'figures': table_expanded_results,
        'text': coref_expanded_results,
    }


# COMPARISON OF CODE CHANGES:

"""
ORIGINAL CODE (sequential):
    figures = self.extract_figures_from_pdf(...)
    images = [figure['figure']['image'] for figure in figures]
    results = self.extract_reactions_from_figures(...)
    table_expanded_results = process_tables(...)
    text_results = self.extract_reactions_from_text_in_pdf(...)
    results_coref = self.extract_molecule_corefs_from_figures_in_pdf(...)
    # ... rest of processing

THREADING VERSION (minimal changes):
    figures = self.extract_figures_from_pdf(...)
    images = [figure['figure']['image'] for figure in figures]

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_reactions = executor.submit(self.extract_reactions_from_figures, ...)
        future_text = executor.submit(self.extract_reactions_from_text_in_pdf, ...)
        future_coref = executor.submit(self.extract_molecule_corefs_from_figures_in_pdf, ...)
        results = future_reactions.result()
        text_results = future_text.result()
        results_coref = future_coref.result()

    # ... rest of processing (same as original)

Lines changed: ~15-20 lines in one method
Complexity: Low
Compatibility: Works with existing code, no model reloading needed
"""


