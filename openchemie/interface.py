import torch
import re
import time
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import layoutparser as lp
import fitz as _fitz
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from molscribe import MolScribe
from rxnscribe import RxnScribe, MolDetect
from chemiener import ChemNER
from .chemrxnextractor import ChemRxnExtractor
from .tableextractor import TableExtractor
from .utils import *
from .timing import time_function_call, log_phase, log_summary, create_timed_wrapper, reset_timing_data, get_timing_data, _get_timing_data

# Suppress harmless checkpoint warning during inference
# This warning appears because checkpointing expects gradients, but during inference
# we use torch.no_grad() which is correct - we don't need gradients for inference
warnings.filterwarnings('ignore', message='.*None of the inputs have requires_grad=True.*')

def _pdf_to_images(pdf_path, last_page=None):
    """Convert PDF pages to PIL images using pymupdf (faster than pdf2image/poppler)."""
    doc = _fitz.open(pdf_path)
    pages = []
    n = last_page if last_page else len(doc)
    zoom = 200 / 72  # match pdf2image default DPI (200)
    mat = _fitz.Matrix(zoom, zoom)
    for i in range(min(n, len(doc))):
        pix = doc[i].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


class OpenChemIE:
    def __init__(self, device=None):
        """
        Initialization function of OpenChemIE
        Parameters:
            device: str of either cuda device name or 'cpu'
        """
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        self._molscribe = None
        self._rxnscribe = None
        self._pdfparser = None
        self._moldet = None
        self._chemrxnextractor = None
        self._chemner = None
        self._coref = None

        # Bbox-based cross-phase caching for MolScribe results
        # Cache key: (figure_id, bbox_tuple) -> {'smiles': str, 'molfile': str, ...}
        self._bbox_cache = {}
        self._bbox_cache_hits = 0
        self._bbox_cache_misses = 0

        # Debug tracking for cache diagnostic
        self._cache_debug = {'stores': [], 'lookups': [], 'iou_checks': []}

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union for two bboxes (normalized coords)."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _bbox_to_tuple(self, bbox):
        """Convert bbox to hashable tuple, rounded for tolerance."""
        return tuple(round(x, 6) for x in bbox)

    def find_cached_smiles(self, figure_id, bbox, iou_threshold=0.5):
        """
        Find cached MolScribe result for a bbox in a figure.

        Args:
            figure_id: Index of the figure being processed
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates
            iou_threshold: Minimum IoU to consider a match (default 0.5)

        Returns:
            Cached prediction dict if found, None otherwise
        """
        bbox_tuple = self._bbox_to_tuple(bbox)

        # Debug: log lookup attempt
        self._cache_debug['lookups'].append({
            'figure_id': figure_id,
            'bbox': list(bbox_tuple),
            'cache_size': len(self._bbox_cache),
            'cache_figure_ids': list(set(fid for fid, _ in self._bbox_cache.keys()))[:10]
        })

        # Try exact match first (fast path)
        key = (figure_id, bbox_tuple)
        if key in self._bbox_cache:
            self._bbox_cache_hits += 1
            return self._bbox_cache[key]

        # Try IoU matching (for cross-detector matches)
        for (fid, cached_bbox), prediction in self._bbox_cache.items():
            if fid == figure_id:
                iou = self._compute_iou(bbox, cached_bbox)
                # Debug: log IoU check
                self._cache_debug['iou_checks'].append({
                    'lookup_fid': figure_id,
                    'cached_fid': fid,
                    'lookup_bbox': list(bbox_tuple),
                    'cached_bbox': list(cached_bbox),
                    'iou': round(iou, 4),
                    'threshold': iou_threshold
                })
                if iou >= iou_threshold:
                    self._bbox_cache_hits += 1
                    return prediction

        self._bbox_cache_misses += 1
        return None

    def cache_smiles(self, figure_id, bbox, prediction):
        """
        Cache a MolScribe prediction result.

        Args:
            figure_id: Index of the figure
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates
            prediction: Dict with 'smiles', 'molfile', etc.
        """
        bbox_tuple = self._bbox_to_tuple(bbox)

        # Debug: log store operation
        self._cache_debug['stores'].append({
            'figure_id': figure_id,
            'bbox': list(bbox_tuple),
            'smiles': prediction.get('smiles', '')[:50] if prediction else ''
        })

        key = (figure_id, bbox_tuple)
        self._bbox_cache[key] = prediction

    def clear_bbox_cache(self):
        """
        Clear the bbox cache. Call between PDFs to limit memory.

        Returns:
            dict: Cache statistics before clearing
        """
        stats = self.get_bbox_cache_stats()
        self._bbox_cache.clear()
        self._bbox_cache_hits = 0
        self._bbox_cache_misses = 0
        return stats

    def get_bbox_cache_stats(self):
        """
        Get bbox cache statistics.

        Returns:
            dict: Contains 'hits', 'misses', 'size', and 'hit_rate'
        """
        total = self._bbox_cache_hits + self._bbox_cache_misses
        hit_rate = self._bbox_cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._bbox_cache_hits,
            'misses': self._bbox_cache_misses,
            'size': len(self._bbox_cache),
            'hit_rate': hit_rate
        }

    def export_cache_debug(self):
        """Export debug data for cache analysis."""
        return self._cache_debug

    def reset_cache_debug(self):
        """Reset debug data for next PDF."""
        self._cache_debug = {'stores': [], 'lookups': [], 'iou_checks': []}

    @property
    def molscribe(self):
        if self._molscribe is None:
            self.init_molscribe()
        return self._molscribe

    @lru_cache(maxsize=None)
    def init_molscribe(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
        self._molscribe = MolScribe(ckpt_path, device=self.device)
        # Attach cache methods to molscribe instance
        self._attach_cache_methods(self._molscribe)

    def _attach_cache_methods(self, molscribe_instance):
        """
        Attach bbox cache methods to a MolScribe instance.
        This enables cross-phase caching when the molscribe is shared.
        """
        import types
        # Bind cache lookup method
        molscribe_instance.find_cached_smiles = types.MethodType(
            lambda self, figure_id, bbox, iou_threshold=0.8: self._openchemie_ref.find_cached_smiles(figure_id, bbox, iou_threshold),
            molscribe_instance
        )
        # Bind cache store method
        molscribe_instance.cache_smiles = types.MethodType(
            lambda self, figure_id, bbox, prediction: self._openchemie_ref.cache_smiles(figure_id, bbox, prediction),
            molscribe_instance
        )
        # Store reference to OpenChemIE for cache access
        molscribe_instance._openchemie_ref = self
        # figure_context will be set by the caller before processing each figure
        molscribe_instance.figure_context = None


    @property
    def rxnscribe(self):
        if self._rxnscribe is None:
            self.init_rxnscribe()
        return self._rxnscribe

    @lru_cache(maxsize=None)
    def init_rxnscribe(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = hf_hub_download("yujieq/RxnScribe", "pix2seq_reaction_full.ckpt")
        # Pass shared molscribe for cross-phase caching
        self._rxnscribe = RxnScribe(ckpt_path, device=self.device, molscribe=self.molscribe)


    @property
    def pdfparser(self):
        if self._pdfparser is None:
            self.init_pdfparser()
        return self._pdfparser

    @lru_cache(maxsize=None)
    def init_pdfparser(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        config_path = "lp://efficientdet/PubLayNet/tf_efficientdet_d1"
        self._pdfparser = lp.AutoLayoutModel(config_path, model_path=ckpt_path, device=self.device.type)


    @property
    def moldet(self):
        if self._moldet is None:
            self.init_moldet()
        return self._moldet

    @lru_cache(maxsize=None)
    def init_moldet(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = hf_hub_download("Ozymandias314/MolDetectCkpt", "best_hf.ckpt")
        # Pass shared molscribe for cross-phase caching
        self._moldet = MolDetect(ckpt_path, device=self.device, molscribe=self.molscribe)


    @property
    def coref(self):
        if self._coref is None:
            self.init_coref()
        return self._coref

    @lru_cache(maxsize=None)
    def init_coref(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = hf_hub_download("Ozymandias314/MolDetectCkpt", "coref_best_hf.ckpt")
        # Pass shared molscribe for cross-phase caching
        self._coref = MolDetect(ckpt_path, device=self.device, coref=True, molscribe=self.molscribe)


    @property
    def chemrxnextractor(self):
        if self._chemrxnextractor is None:
            self.init_chemrxnextractor()
        return self._chemrxnextractor

    @lru_cache(maxsize=None)
    def init_chemrxnextractor(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = snapshot_download(repo_id="amberwang/chemrxnextractor-training-modules")
        self._chemrxnextractor = ChemRxnExtractor("", None, ckpt_path, self.device.type)


    @property
    def chemner(self):
        if self._chemner is None:
            self.init_chemner()
        return self._chemner

    @lru_cache(maxsize=None)
    def init_chemner(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = hf_hub_download("Ozymandias314/ChemNERckpt", "best.ckpt")
        self._chemner = ChemNER(ckpt_path, device=self.device)


    @property
    def tableextractor(self):
        return TableExtractor()


    def extract_figures_from_pdf(self, pdf, num_pages=None, output_bbox=False, output_image=True):
        """
        Find and return all figures from a pdf page
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
            output_bbox: whether to output bounding boxes for each individual entry of a table
            output_image: whether to include PIL image for figures. default is True
        Returns:
            list of content in the following format
            [
                { # first figure
                    'title': str,
                    'figure': {
                        'image': PIL image or None,
                        'bbox': list in form [x1, y1, x2, y2],
                    }
                    'table': {
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                        'content': {
                            'columns': list of column headers,
                            'rows': list of list of row content,
                        } or None
                    }
                    'footnote': str or empty,
                    'page': int
                }
                # more figures
            ]
        """
        pages = _pdf_to_images(pdf, last_page=num_pages)

        table_ext = self.tableextractor
        table_ext.set_pdf_file(pdf)
        table_ext.set_output_image(output_image)

        table_ext.set_output_bbox(output_bbox)

        return table_ext.extract_all_tables_and_figures(pages, self.pdfparser, content='figures')

    def extract_tables_from_pdf(self, pdf, num_pages=None, output_bbox=False, output_image=True):
        """
        Find and return all tables from a pdf page
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
            output_bbox: whether to include bboxes for individual entries of the table
            output_image: whether to include PIL image for figures. default is True
        Returns:
            list of content in the following format
            [
                { # first table
                    'title': str,
                    'figure': {
                        'image': PIL image or None,
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                    }
                    'table': {
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                        'content': {
                            'columns': list of column headers,
                            'rows': list of list of row content,
                        }
                    }
                    'footnote': str or empty,
                    'page': int
                }
                # more tables
            ]
        """
        pages = _pdf_to_images(pdf, last_page=num_pages)

        table_ext = self.tableextractor
        table_ext.set_pdf_file(pdf)
        table_ext.set_output_image(output_image)

        table_ext.set_output_bbox(output_bbox)

        return table_ext.extract_all_tables_and_figures(pages, self.pdfparser, content='tables')

    def extract_figures_and_tables_from_pdf(self, pdf, num_pages=None, output_bbox=False, output_image=True):
        """
        Extract both figures and tables in a single pass, sharing PDF-to-image
        conversion, LayoutParser inference, and pdfminer page caching.

        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
            output_bbox: whether to output bounding boxes for each individual entry of a table
            output_image: whether to include PIL image for figures. default is True

        Returns:
            tuple of (figures, tables) where each is a list of dicts in the same
            format as extract_figures_from_pdf / extract_tables_from_pdf
        """
        import time as _time
        t_render = _time.perf_counter()
        pages = _pdf_to_images(pdf, last_page=num_pages)
        render_time = _time.perf_counter() - t_render

        table_ext = self.tableextractor
        table_ext.set_pdf_file(pdf)
        table_ext.set_output_image(output_image)
        table_ext.set_output_bbox(output_bbox)

        all_results = table_ext.extract_all_tables_and_figures(pages, self.pdfparser, content=None)

        # Expose sub-phase timing from table extractor, include render time
        timing = getattr(table_ext, '_last_timing', None)
        if timing is not None:
            timing['pdf_render'] = render_time
        self._last_fig_table_timing = timing

        figures = []
        tables = []
        for item in all_results:
            if item.get('table', {}).get('content') is not None:
                tables.append(item)
            else:
                figures.append(item)
        return figures, tables

    def extract_molecules_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None, skip_molblock=False):
        """
        Get all molecules and their information from a pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        reset_timing_data()
        total_start = time.time()

        figures = time_function_call(
            self.extract_figures_from_pdf,
            pdf, num_pages=num_pages, output_bbox=True,
            module_name="extract_figures_from_pdf",
            silent=True
        )
        images = [figure['figure']['image'] for figure in figures]
        results = time_function_call(
            self.extract_molecules_from_figures,
            images, batch_size, skip_molblock,
            module_name="extract_molecules_from_figures",
            silent=True
        )
        for figure, result in zip(figures, results):
            result['page'] = figure['page']

        total_time = time.time() - total_start
        timing_data = get_timing_data()
        timing_data['total_time'] = total_time
        timing_data['num_figures'] = len(figures)

        # Add timing to first result if available
        if results:
            results[0]['_timing'] = timing_data

        return results

    def extract_molecule_bboxes_from_figures(self, figures, batch_size=16):
        """
        Return bounding boxes of molecules in images
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                [   # first figure
                    {   # first bounding box
                        'category': str,
                        'bbox': tuple in the form (x1, y1, x2, y2),
                        'category_id': int,
                        'score': float
                    },
                    # more bounding boxes
                ],
                # more figures
            ]
        """
        figures = [convert_to_pil(figure) for figure in figures]
        return self.moldet.predict_images(figures, batch_size=batch_size)

    def extract_molecules_from_figures(self, figures, batch_size=16, skip_molblock=False):
        """
        Get all molecules and their information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                },
                # more figures
            ]
        """
        from .timing import time_module, _get_timing_data, reset_timing_data

        reset_timing_data()
        total_start = time.time()

        with time_module("moldet.predict_images", silent=True):
            bboxes = self.extract_molecule_bboxes_from_figures(figures, batch_size=batch_size)

        # Capture MolDetect's detailed timing
        moldet_timing = None
        if hasattr(self.moldet, 'get_last_timing'):
            moldet_timing = self.moldet.get_last_timing()

        with time_module("convert_to_cv2", silent=True):
            figures = [convert_to_cv2(figure) for figure in figures]

        with time_module("clean_bbox_output", silent=True):
            results, cropped_images, refs = clean_bbox_output(figures, bboxes)

        with time_module("molscribe.predict_images", silent=True):
            mol_info = self.molscribe.predict_images(cropped_images, batch_size=batch_size, skip_molblock=skip_molblock, return_atoms_bonds=True)

        # Capture MolScribe's detailed timing
        molscribe_timing = None
        if hasattr(self.molscribe, 'get_last_timing'):
            molscribe_timing = self.molscribe.get_last_timing()

        for info, ref in zip(mol_info, refs):
            ref.update(info)

        # Populate cache with molecule extraction results for cross-phase reuse
        # This enables reaction extraction to reuse molecules found during molecule extraction
        for fig_idx, result in enumerate(results):
            for mol in result.get('molecules', []):
                bbox = mol.get('bbox')
                if bbox and mol.get('smiles'):
                    self.cache_smiles(fig_idx, bbox, {
                        'smiles': mol.get('smiles'),
                        'molfile': mol.get('molfile'),
                        'atoms': mol.get('atoms'),
                        'bonds': mol.get('bonds')
                    })

        # Add timing data to results
        timing_data = _get_timing_data()
        timing_data['total_time'] = time.time() - total_start
        timing_data['num_figures'] = len(results)
        timing_data['num_molecules'] = sum(len(r.get('molecules', [])) for r in results)

        # Include MolDetect's detailed timing breakdown
        if moldet_timing:
            timing_data['moldet_breakdown'] = moldet_timing

        # Include MolScribe's detailed timing breakdown
        if molscribe_timing:
            timing_data['molscribe_breakdown'] = molscribe_timing

        # Attach timing to first result for retrieval
        if results:
            results[0]['_timing'] = timing_data.copy()

        return results

    def extract_molecule_corefs_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None, molscribe = True, ocr = True):
        """
        Get all molecule bboxes and corefs from figures in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of results for each figure in the following format:
            [
                {
                    'bboxes': [
                        {   # first bbox
                            'category': '[Sup]',
                            'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168),
                            'category_id': 4,
                            'score': -0.07593922317028046
                        },
                        # More bounding boxes
                    ],
                    'corefs': [
                        [0, 1],  # molecule bbox index, identifier bbox index
                        [3, 4],
                        # More coref pairs
                    ],
                    'page': int
                },
                # More figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_molecule_corefs_from_figures(images, batch_size=batch_size, molscribe=molscribe, ocr=ocr)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results

    def extract_molecule_corefs_from_figures(self, figures, batch_size=16, molscribe=True, ocr=True):
        """
        Get all molecule bboxes and corefs from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format:
            [
                {
                    'bboxes': [
                        {   # first bbox
                            'category': '[Sup]',
                            'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168),
                            'category_id': 4,
                            'score': -0.07593922317028046
                        },
                        # More bounding boxes
                    ],
                    'corefs': [
                        [0, 1],  # molecule bbox index, identifier bbox index
                        [3, 4],
                        # More coref pairs
                    ],
                },
                # More figures
            ]
        """
        figures = [convert_to_pil(figure) for figure in figures]
        return self.coref.predict_images(figures, batch_size=batch_size, coref=True, molscribe = molscribe, ocr = ocr)

    def extract_reactions_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None, molscribe=True, ocr=True, skip_molblock=False):
        """
        Get reaction information from figures in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
            molscribe: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        reset_timing_data()
        total_start = time.time()

        figures = time_function_call(
            self.extract_figures_from_pdf,
            pdf, num_pages=num_pages, output_bbox=True,
            module_name="extract_figures_from_pdf",
            silent=True
        )
        images = [figure['figure']['image'] for figure in figures]
        results = time_function_call(
            self.extract_reactions_from_figures,
            images, batch_size, molscribe, ocr, skip_molblock,
            module_name="extract_reactions_from_figures",
            silent=True
        )
        for figure, result in zip(figures, results):
            result['page'] = figure['page']

        total_time = time.time() - total_start
        timing_data = get_timing_data()
        timing_data['total_time'] = total_time
        timing_data['num_figures'] = len(figures)

        # Add timing to first result if available
        if results:
            results[0]['_timing'] = timing_data

        return results

    def extract_reactions_from_figures(self, figures, batch_size=16, molscribe=True, ocr=True, skip_molblock=False):
        """
        Get reaction information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference in all models
            molscribe: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                },
                # more figures
            ]

        """
        from .timing import time_module, _get_timing_data, reset_timing_data

        reset_timing_data()
        total_start = time.time()

        with time_module("convert_to_pil", silent=True):
            pil_figures = [convert_to_pil(figure) for figure in figures]

        results = []
        with time_module("rxnscribe.predict_images", silent=True):
            reactions = self.rxnscribe.predict_images(pil_figures, batch_size=batch_size, molscribe=molscribe, ocr=ocr, skip_molblock=skip_molblock)

        # Capture RxnScribe's detailed timing
        rxnscribe_timing = None
        if hasattr(self.rxnscribe, 'get_last_timing'):
            rxnscribe_timing = self.rxnscribe.get_last_timing()

        for figure, rxn in zip(figures, reactions):
            data = {
                'figure': figure,
                'reactions': rxn,
                }
            results.append(data)

        # Add timing data to results
        timing_data = _get_timing_data()
        timing_data['total_time'] = time.time() - total_start
        timing_data['num_figures'] = len(results)
        timing_data['num_reactions'] = sum(len(r.get('reactions', [])) for r in results)

        # Include RxnScribe's detailed timing breakdown
        if rxnscribe_timing:
            timing_data['rxnscribe_breakdown'] = rxnscribe_timing

        # Attach timing to first result for retrieval
        if results:
            results[0]['_timing'] = timing_data.copy()

        return results

    def extract_molecules_from_text_in_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Get molecules in text of given pdf

        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of sentences and found molecules in the following format
            [
                {
                    'molecules': [
                        { # first paragraph
                            'text': str,
                            'labels': [
                                (str, int, int), # tuple of label, range start (inclusive), range end (exclusive)
                                # more labels
                            ]
                        },
                        # more paragraphs
                    ]
                    'page': int
                },
                # more pages
            ]
        """
        self.chemrxnextractor.set_pdf_file(pdf)
        self.chemrxnextractor.set_pages(num_pages)
        text = self.chemrxnextractor.get_paragraphs_from_pdf(num_pages)
        result = []
        for data in text:
            model_inp = []
            for paragraph in data['paragraphs']:
                model_inp.append(' '.join(paragraph).replace('\n', ''))
            output = self.chemner.predict_strings(model_inp, batch_size=batch_size)
            to_add = {
                'molecules': [{
                    'text': t,
                    'labels': labels,
                    } for t, labels in zip(model_inp, output)],
                'page': data['page']
            }
            result.append(to_add)
        return result


    def extract_reactions_from_text_in_pdf(self, pdf, num_pages=None):
        """
        Get reaction information from text in pdf
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of pages and corresponding reaction info in the following format
            [
                {
                    'page': page number
                    'reactions': [
                        {
                            'tokens': list of words in relevant sentence,
                            'reactions' : [
                                {
                                    # key, value pairs where key is the label and value is a tuple
                                    # or list of tuples of the form (tokens, start index, end index)
                                    # where indices are for the corresponding token list and start and end are inclusive
                                }
                                # more reactions
                            ]
                        }
                        # more reactions in other sentences
                    ]
                },
                # more pages
            ]
        """
        self.chemrxnextractor.set_pdf_file(pdf)
        self.chemrxnextractor.set_pages(num_pages)
        return self.chemrxnextractor.extract_reactions_from_text()

    def extract_molecules_and_reactions_from_text_in_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Combined text extraction: molecules + reactions from a single PDF parse.
        Avoids duplicate pdftotext + paragraph splitting.

        Returns:
            tuple: (molecule_results, reaction_results)
        """
        self.chemrxnextractor.set_pdf_file(pdf)
        self.chemrxnextractor.set_pages(num_pages)
        text = self.chemrxnextractor.get_paragraphs_from_pdf(num_pages)

        # ChemNER for molecules
        mol_result = []
        for data in text:
            model_inp = []
            for paragraph in data['paragraphs']:
                model_inp.append(' '.join(paragraph).replace('\n', ''))
            output = self.chemner.predict_strings(model_inp, batch_size=batch_size)
            mol_result.append({
                'molecules': [{'text': t, 'labels': labels} for t, labels in zip(model_inp, output)],
                'page': data['page']
            })

        # ChemRxnExtractor for reactions (reuses already-parsed text)
        rxn_result = []
        for data in text:
            L = [sent for paragraph in data['paragraphs'] for sent in paragraph]
            reactions = self.chemrxnextractor.get_reactions(L, page_number=data['page'])
            rxn_result.append(reactions)

        return mol_result, rxn_result

    def preparse_text_from_pdfs(self, pdf_paths, num_pages=None):
        """
        Pre-parse text from multiple PDFs. Returns parsed paragraph data
        that can be fed to mega_batch_text_extraction().

        Args:
            pdf_paths: List of PDF file paths
            num_pages: Optional limit on number of pages to process per PDF

        Returns:
            List of (pdf_idx, text_data) tuples where text_data is the paragraph structure
        """
        all_preparsed = []
        for pdf_idx, pdf_path in enumerate(pdf_paths):
            self.chemrxnextractor.set_pdf_file(pdf_path)
            self.chemrxnextractor.set_pages(num_pages)
            text = self.chemrxnextractor.get_paragraphs_from_pdf(num_pages)
            all_preparsed.append((pdf_idx, text))
        return all_preparsed

    def mega_batch_text_extraction(self, preparsed_data, batch_size=16):
        """
        Run ChemNER and ChemRxnExtractor on pre-parsed text from multiple PDFs
        in single mega-batches for maximum GPU utilization.

        Args:
            preparsed_data: Output from preparse_text_from_pdfs()
            batch_size: Batch size for ChemNER inference

        Returns:
            Dict mapping pdf_idx -> (molecule_results, reaction_results)
        """
        # Collect all paragraphs across all PDFs for mega-batch ChemNER
        all_strings = []
        boundaries = []  # (pdf_idx, page, string_start, string_end)

        for pdf_idx, text_pages in preparsed_data:
            for data in text_pages:
                start = len(all_strings)
                for paragraph in data['paragraphs']:
                    all_strings.append(' '.join(paragraph).replace('\n', ''))
                boundaries.append((pdf_idx, data['page'], start, len(all_strings)))

        # Single mega-batch ChemNER call
        all_ner_output = self.chemner.predict_strings(all_strings, batch_size=batch_size) if all_strings else []

        # Collect all sentences for mega-batch ChemRxnExtractor
        all_sentences = []
        rxn_boundaries = []  # (pdf_idx, page, sent_start, sent_end)

        for pdf_idx, text_pages in preparsed_data:
            for data in text_pages:
                start = len(all_sentences)
                for paragraph in data['paragraphs']:
                    all_sentences.extend(paragraph)
                rxn_boundaries.append((pdf_idx, data['page'], start, len(all_sentences)))

        # Single mega-batch ChemRxnExtractor call
        all_rxn_output = self.chemrxnextractor.rxn_extractor.get_reactions(all_sentences) if all_sentences else []

        # Distribute results back to per-PDF structure
        results = {}
        # Build molecule results
        for pdf_idx, page, s_start, s_end in boundaries:
            if pdf_idx not in results:
                results[pdf_idx] = ([], [])
            page_strings = all_strings[s_start:s_end]
            page_ner = all_ner_output[s_start:s_end]
            results[pdf_idx][0].append({
                'molecules': [{'text': t, 'labels': labels} for t, labels in zip(page_strings, page_ner)],
                'page': page
            })

        # Build reaction results
        for pdf_idx, page, sent_start, sent_end in rxn_boundaries:
            if pdf_idx not in results:
                results[pdf_idx] = ([], [])
            page_rxns = all_rxn_output[sent_start:sent_end]
            ret = [r for r in page_rxns if len(r.get('reactions', [])) != 0]
            results[pdf_idx][1].append({'page': page, 'reactions': ret})

        return results

    def extract_reactions_from_text_in_pdf_combined(self, pdf, num_pages=None):
        """
        Get reaction information from text in pdf and combined with corefs from figures
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of pages and corresponding reaction info in the following format
            [
                {
                    'page': page number
                    'reactions': [
                        {
                            'tokens': list of words in relevant sentence,
                            'reactions' : [
                                {
                                    # key, value pairs where key is the label and value is a tuple
                                    # or list of tuples of the form (tokens, start index, end index)
                                    # where indices are for the corresponding token list and start and end are inclusive
                                }
                                # more reactions
                            ]
                        }
                        # more reactions in other sentences
                    ]
                },
                # more pages
            ]
        """
        results = self.extract_reactions_from_text_in_pdf(pdf, num_pages=num_pages)
        results_coref = self.extract_molecule_corefs_from_figures_in_pdf(pdf, num_pages=num_pages)
        return associate_corefs(results, results_coref)

    def extract_reactions_from_figures_and_tables_in_pdf(self, pdf, num_pages=None, batch_size=16, molscribe=True, ocr=True, skip_molblock=False):
        """
        Get reaction information from figures and combine with table information in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
            molscribe: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_reactions_from_figures(images, batch_size=batch_size, molscribe=molscribe, ocr=ocr, skip_molblock=skip_molblock)
        
        # Reset R-group timing before R-group processing
        reset_rgroup_timing()
        
        results = process_tables(figures, results, self.molscribe, batch_size=batch_size)
        results_coref = self.extract_molecule_corefs_from_figures_in_pdf(pdf, num_pages=num_pages)
        results = replace_rgroups_in_figure(figures, results, results_coref, self.molscribe, batch_size=batch_size)
        results = expand_reactions_with_backout(results, results_coref, self.molscribe)
        
        # Capture R-group timing and attach to results
        rgroup_timing = get_rgroup_timing()
        if rgroup_timing and rgroup_timing.get('molscribe_calls') and results:
            results[0]['_rgroup_timing'] = rgroup_timing
        
        return results

    def extract_reactions_from_pdf(self, pdf, num_pages=None, batch_size=16):
        """
        Returns:
            dictionary of reactions from multimodal sources
            {
                'figures': [
                    {
                        'figure': PIL image
                        'reactions': [
                            {
                                'reactants': [
                                    {
                                        'category': str,
                                        'bbox': tuple (x1,x2,y1,y2),
                                        'category_id': int,
                                        'smiles': str,
                                        'molfile': str,
                                    },
                                    # more reactants
                                ],
                                'conditions': [
                                    {
                                        'category': str,
                                        'text': list of str,
                                    },
                                    # more conditions
                                ],
                                'products': [
                                    # same structure as reactants
                                ]
                            },
                            # more reactions
                        ],
                        'page': int
                    },
                    # more figures
                ]
                'text': [
                    {
                        'page': page number
                        'reactions': [
                            {
                                'tokens': list of words in relevant sentence,
                                'reactions' : [
                                    {
                                        # key, value pairs where key is the label and value is a tuple
                                        # or list of tuples of the form (tokens, start index, end index)
                                        # where indices are for the corresponding token list and start and end are inclusive
                                    }
                                    # more reactions
                                ]
                            }
                            # more reactions in other sentences
                        ]
                    },
                    # more pages
                ]
            }

        """
        # Reset timing data for this run
        reset_timing_data()
        total_start = time.time()

        # Phase 1: Extract figures first (needed for later steps)
        log_phase("Phase 1: Extract figures", silent=True)
        figures = time_function_call(
            self.extract_figures_from_pdf,
            pdf, num_pages=num_pages, output_bbox=True,
            module_name="extract_figures_from_pdf",
            silent=True
        )
        images = [figure['figure']['image'] for figure in figures]

        # Phase 2: Launch independent operations in parallel
        log_phase("Phase 2: Parallel operations", silent=True)
        # These can run concurrently since they don't depend on each other:
        # - extract_reactions_from_figures: processes figure images
        # - extract_reactions_from_text_in_pdf: processes PDF text (independent)
        # - extract_molecule_corefs_from_figures_in_pdf: processes figures again (independent)
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Create timed wrappers for each function (silent to avoid spam)
            timed_extract_reactions = create_timed_wrapper(
                self.extract_reactions_from_figures,
                "extract_reactions_from_figures",
                silent=True
            )
            timed_extract_text = create_timed_wrapper(
                self.extract_reactions_from_text_in_pdf,
                "extract_reactions_from_text_in_pdf",
                silent=True
            )
            timed_extract_coref = create_timed_wrapper(
                self.extract_molecule_corefs_from_figures_in_pdf,
                "extract_molecule_corefs_from_figures_in_pdf",
                silent=True
            )

            future_reactions = executor.submit(
                timed_extract_reactions,
                images, batch_size, True, True
            )
            future_text = executor.submit(
                timed_extract_text,
                pdf, num_pages
            )
            future_coref = executor.submit(
                timed_extract_coref,
                pdf, batch_size, num_pages
            )

            # Wait for all parallel operations to complete and collect timing
            reactions_result, reactions_timing = future_reactions.result()
            text_result, text_timing = future_text.result()
            coref_result, coref_timing = future_coref.result()

            # Store timing data from parallel operations
            timing_data = _get_timing_data()
            timing_data['modules'].append(reactions_timing)
            timing_data['modules'].append(text_timing)
            timing_data['modules'].append(coref_timing)

            results = reactions_result
            text_results = text_result
            results_coref = coref_result

        # Phase 3: Sequential post-processing (dependencies must be resolved)
        log_phase("Phase 3: Post-processing", silent=True)
        
        # Reset R-group timing before R-group processing
        reset_rgroup_timing()
        
        table_expanded_results = time_function_call(
            process_tables,
            figures, results, self.molscribe, batch_size,
            module_name="process_tables",
            silent=True
        )
        figure_results = time_function_call(
            replace_rgroups_in_figure,
            figures, table_expanded_results, results_coref, self.molscribe, batch_size,
            module_name="replace_rgroups_in_figure",
            silent=True
        )
        table_expanded_results = time_function_call(
            expand_reactions_with_backout,
            figure_results, results_coref, self.molscribe,
            module_name="expand_reactions_with_backout",
            silent=True
        )
        
        # Capture R-group MolScribe timing
        rgroup_timing = get_rgroup_timing()
        
        coref_expanded_results = time_function_call(
            associate_corefs,
            text_results, results_coref,
            module_name="associate_corefs",
            silent=True
        )

        total_time = time.time() - total_start
        log_summary(total_time, num_figures=len(figures), num_pages=num_pages, silent=True)

        # Get timing data for return
        timing_data = get_timing_data()
        
        # Include R-group MolScribe timing if available
        if rgroup_timing and rgroup_timing.get('molscribe_calls'):
            timing_data['rgroup_molscribe_timing'] = rgroup_timing

        return {
            'figures': table_expanded_results,
            'text': coref_expanded_results,
            '_timing': timing_data  # Include timing data in return
        }

if __name__=="__main__":
    model = OpenChemIE()
