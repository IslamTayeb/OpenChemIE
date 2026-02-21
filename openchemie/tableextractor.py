import time
import numpy as np
from PIL import Image
import layoutparser as lp
import fitz

from PyPDF2 import PdfReader
import pandas as pd

from operator import itemgetter

# inputs: pdf_file, page #, bounding box (optional) (llur or ullr), output_bbox
class TableExtractor(object):
    def __init__(self, output_bbox=True):
        self.pdf_file = ""
        self.page = ""
        self.image_dpi = 200
        self.pdf_dpi = 72
        self.output_bbox = output_bbox
        self.blocks = {}
        self.title_y = 0
        self.column_header_y = 0
        self.model = None
        self.img = None
        self.output_image = True
        self._page_layout_cache = {}
        self.tagging = {
            'substance': ['compound', 'salt', 'base', 'solvent', 'CBr4', 'collidine', 'InX3', 'substrate', 'ligand', 'PPh3', 'PdL2', 'Cu', 'compd', 'reagent', 'reagant', 'acid', 'aldehyde', 'amine', 'Ln', 'H2O', 'enzyme', 'cofactor', 'oxidant', 'Pt(COD)Cl2', 'CuBr2', 'additive'],
            'ratio': [':'],
            'measurement': ['μM', 'nM', 'IC50', 'CI', 'excitation', 'emission', 'Φ', 'φ', 'shift', 'ee', 'ΔG', 'ΔH', 'TΔS', 'Δ', 'distance', 'trajectory', 'V', 'eV'],
            'temperature': ['temp', 'temperature', 'T', '°C'],
            'time': ['time', 't(', 't ('],
            'result': ['yield', 'aa', 'result', 'product', 'conversion', '(%)'],
            'alkyl group': ['R', 'Ar', 'X', 'Y'],
            'solvent': ['solvent'],
            'counter': ['entry', 'no.'],
            'catalyst': ['catalyst', 'cat.'],
            'conditions': ['condition'],
            'reactant': ['reactant'],
        }
        
    def set_output_image(self, oi):
        self.output_image = oi
    
    def set_pdf_file(self, pdf):
        if pdf != self.pdf_file:
            self._page_layout_cache = {}
        self.pdf_file = pdf
    
    def set_page_num(self, pn):
        self.page = pn
        
    def set_output_bbox(self, ob):
        self.output_bbox = ob
        
    def _get_page_text(self, page_num):
        """Return (lines, blocks) with bboxes in PDF coords (bottom-left origin).
        lines: list of [x0, y0, x1, y1, text] for individual text lines
        blocks: list of [x0, y0, x1, y1, text] for text blocks
        """
        if page_num not in self._page_layout_cache:
            doc = fitz.open(self.pdf_file)
            page = doc[page_num]
            page_height = page.rect.height
            lines = []
            blocks = []
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                bb = block["bbox"]
                block_text = ""
                for line in block["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    lb = line["bbox"]
                    # Convert from top-left origin (pymupdf) to bottom-left origin (PDF)
                    lines.append([lb[0], page_height - lb[3], lb[2], page_height - lb[1], text])
                    block_text += text + "\n"
                blocks.append([bb[0], page_height - bb[3], bb[2], page_height - bb[1], block_text])
            doc.close()
            self._page_layout_cache[page_num] = (lines, blocks)
        return self._page_layout_cache[page_num]

    # type is what coordinates you want to get. it comes in text, title, list, table, and figure
    def convert_to_pdf_coordinates(self, type):
        # scale coordinates

        blocks = self.blocks[type]
        coordinates =  [blocks[a].scale(self.pdf_dpi/self.image_dpi) for a in range(len(blocks))]

        reader = PdfReader(self.pdf_file)
        p = reader.pages[self.page]
        a = p.mediabox.upper_left
        new_coords = []
        for new_block in coordinates:
            new_coords.append((new_block.block.x_1, pd.to_numeric(a[1]) - new_block.block.y_2, new_block.block.x_2, pd.to_numeric(a[1]) - new_block.block.y_1))

        return new_coords
    # output: list of bounding boxes for tables but in pdf coordinates
    
    # input: new_coords is singular table bounding box in pdf coordinates
    def extract_singular_table(self, new_coords):
        lines, _ = self._get_page_text(self.page)
        elements = []
        x_min, x_max = min(new_coords[0], new_coords[2]), max(new_coords[0], new_coords[2])
        y_min, y_max = min(new_coords[1], new_coords[3]), max(new_coords[1], new_coords[3])
        for line in lines:
            lx0, ly0, lx1, ly1 = line[:4]
            if (lx0 > x_min and lx0 < x_max and ly0 > y_min and ly0 < y_max and
                    lx1 > x_min and lx1 < x_max and ly1 > y_min and ly1 < y_max):
                elements.append([lx0, ly0, lx1, ly1, line[4]])

        elements = sorted(elements, key=itemgetter(0))
        w = sorted(elements, key=itemgetter(3), reverse=True)
        if len(w) <= 1:
            return None

        ret = {}
        i = 1
        g = [w[0]]

        while i < len(w) and w[i][3] > w[i-1][1]:
            g.append(w[i])
            i += 1
        g = sorted(g, key=itemgetter(0))
        # check for overlaps
        for a in range(len(g)-1, 0, -1):
            if g[a][0] < g[a-1][2]:
                g[a-1][0] = min(g[a][0], g[a-1][0])
                g[a-1][1] = min(g[a][1], g[a-1][1])
                g[a-1][2] = max(g[a][2], g[a-1][2])
                g[a-1][3] = max(g[a][3], g[a-1][3])
                g[a-1][4] = g[a-1][4].strip() + " " + g[a][4]
                g.pop(a)


        ret.update({"columns":[]})
        for t in g:
            temp_bbox = t[:4]

            column_text = t[4].strip()
            tag = 'unknown'
            tagged = False
            for key in self.tagging.keys():
                for word in self.tagging[key]:
                    if word in column_text:
                        tag = key
                        tagged = True
                        break
                if tagged:
                    break

            if self.output_bbox:
                ret["columns"].append({'text':column_text,'tag': tag, 'bbox':temp_bbox})
            else:
                ret["columns"].append({'text':column_text,'tag': tag})
            self.column_header_y = max(t[1], t[3])
        ret.update({"rows":[]})

        g.insert(0, [0, 0, new_coords[0], 0, ''])
        g.append([new_coords[2], 0, 0, 0, ''])
        while i < len(w):
            group = [w[i]]
            i += 1
            while i < len(w) and w[i][3] > w[i-1][1]:
                group.append(w[i])
                i += 1
            group = sorted(group, key=itemgetter(0))

            for a in range(len(group)-1, 0, -1):
                if group[a][0] < group[a-1][2]:
                    group[a-1][0] = min(group[a][0], group[a-1][0])
                    group[a-1][1] = min(group[a][1], group[a-1][1])
                    group[a-1][2] = max(group[a][2], group[a-1][2])
                    group[a-1][3] = max(group[a][3], group[a-1][3])
                    group[a-1][4] = group[a-1][4].strip() + " " + group[a][4]
                    group.pop(a)

            a = 1
            while a < len(g) - 1:
                if a > len(group):
                    group.append([0, 0, 0, 0, '\n'])
                    a += 1
                    continue
                if group[a-1][0] >= g[a-1][2] and group[a-1][2] <= g[a+1][0]:
                    pass
                    """
                    if a < len(group) and group[a][0] >= g[a-1][2] and group[a][2] <= g[a+1][0]:
                        g.insert(1, [g[0][2], 0, group[a-1][2], 0, ''])
                        #ret["columns"].insert(0, '')
                    else:
                        a += 1
                        continue
                    """
                else: group.insert(a-1, [0, 0, 0, 0, '\n'])
                a += 1


            added_row = []
            for t in group:
                temp_bbox = t[:4]
                if self.output_bbox:
                    added_row.append({'text':t[4].strip(), 'bbox':temp_bbox})
                else:
                    added_row.append(t[4].strip())
            ret["rows"].append(added_row)
        if ret["rows"] and len(ret["rows"][0]) != len(ret["columns"]):
            ret["columns"] = ret["rows"][0]
            ret["rows"] = ret["rows"][1:]
            for col in ret['columns']:
                tag = 'unknown'
                tagged = False
                for key in self.tagging.keys():
                    for word in self.tagging[key]:
                        if word in col['text']:
                            tag = key
                            tagged = True
                            break
                    if tagged:
                        break
                col['tag'] = tag

        return ret
            
    def get_title_and_footnotes(self, tb_coords):
        _, blocks = self._get_page_text(self.page)
        title = (0, 0, 0, 0, '')
        footnote = (0, 0, 0, 0, '')
        title_gap = 30
        footnote_gap = 30
        for block in blocks:
            bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
            # Check x-overlap with table coordinates
            if (bx0 >= tb_coords[0] and bx0 <= tb_coords[2]) or (bx1 >= tb_coords[0] and bx1 <= tb_coords[2]) or (tb_coords[0] >= bx0 and tb_coords[0] <= bx1) or (tb_coords[2] >= bx0 and tb_coords[2] <= bx1):
                if 'Table' in text:
                    if abs(by0 - tb_coords[3]) < title_gap:
                        title = (bx0, by0, bx1, by1, text[text.index('Table'):].replace('\n', ' '))
                        title_gap = abs(by0 - tb_coords[3])
                if 'Scheme' in text:
                    if abs(by0 - tb_coords[3]) < title_gap:
                        title = (bx0, by0, bx1, by1, text[text.index('Scheme'):].replace('\n', ' '))
                        title_gap = abs(by0 - tb_coords[3])
                if by0 >= tb_coords[1] and by1 <= tb_coords[3]: continue
                temp = ['aA', 'aB', 'aC', 'aD', 'aE', 'aF', 'aG', 'aH', 'aI', 'aJ', 'aK', 'aL', 'aM', 'aN', 'aO', 'aP', 'aQ', 'aR', 'aS', 'aT', 'aU', 'aV', 'aW', 'aX', 'aY', 'aZ', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a0']
                for segment in temp:
                    if segment in text:
                        if abs(by1 - tb_coords[1]) < footnote_gap:
                            footnote = (bx0, by0, bx1, by1, text[text.index(segment):].replace('\n', ' '))
                            footnote_gap = abs(by1 - tb_coords[1])
                        break
        self.title_y = min(title[1], title[3])
        if self.output_bbox:
            return ({'text': title[4], 'bbox': list(title[:4])}, {'text': footnote[4], 'bbox': list(footnote[:4])})
        else:
            return (title[4], footnote[4])
            
    def extract_table_information(self):
        table_coordinates = self.blocks['table'] #should return a list of layout objects
        table_coordinates_in_pdf = self.convert_to_pdf_coordinates('table') #should return a list of lists

        ans = []
        i = 0
        for coordinate in table_coordinates_in_pdf:
            ret = {}
            pad = 20
            coordinate = [coordinate[0] - pad, coordinate[1], coordinate[2] + pad, coordinate[3]]
            ullr_coord = [coordinate[0], coordinate[3], coordinate[2], coordinate[1]]
        
            table_results = self.extract_singular_table(coordinate)
            tf = self.get_title_and_footnotes(coordinate)
            figure = Image.fromarray(table_coordinates[i].crop_image(self.img))
            ret.update({'title': tf[0]})
            ret.update({'figure': {
                'image': None,
                'bbox': []
                       }})
            if self.output_image:
                ret['figure']['image'] = figure
            ret.update({'table': {'bbox': list(coordinate), 'content': table_results}})
            ret.update({'footnote': tf[1]})
            if abs(self.title_y - self.column_header_y) > 50:
                ret['figure']['bbox'] = list(coordinate)
            
            ret.update({'page':self.page})
            
            ans.append(ret)
            i += 1
        
        return ans
        
    def extract_figure_information(self):
        figure_coordinates = self.blocks['figure']
        figure_coordinates_in_pdf = self.convert_to_pdf_coordinates('figure')
        
        ans = []
        for i in range(len(figure_coordinates)):
            ret = {}
            coordinate = figure_coordinates_in_pdf[i]
            ullr_coord = [coordinate[0], coordinate[3], coordinate[2], coordinate[1]]
            
            tf = self.get_title_and_footnotes(coordinate)
            figure = Image.fromarray(figure_coordinates[i].crop_image(self.img))
            ret.update({'title':tf[0]})
            ret.update({'figure': {
                'image': None,
                'bbox': []
                       }})
            if self.output_image:
                ret['figure']['image'] = figure
            ret.update({'table': {
                'bbox': [],
                'content': None
                       }})
            ret.update({'footnote': tf[1]})
            ret['figure']['bbox'] = list(coordinate)
                
            ret.update({'page':self.page})
            
            ans.append(ret)
        
        return ans
            
        
    def _detect_batched(self, pages, page_indices, pdfparser):
        """Run LayoutParser on multiple pages in a single batched forward pass.

        Args:
            pages: List of PIL images
            page_indices: List of page index for each image
            pdfparser: LayoutParser EfficientDet model

        Returns:
            Dict mapping page_index -> Layout result
        """
        import torch

        # Preprocess all pages
        all_inputs = []
        all_infos = []
        for page in pages:
            pil_img = pdfparser.image_loader(np.asarray(page))
            model_input, image_info = pdfparser.preprocessor.preprocess(pil_img)
            all_inputs.append(model_input)
            all_infos.append(image_info)

        # Batched forward pass (chunked to limit GPU memory)
        chunk_size = 32
        all_outputs = []
        for start in range(0, len(all_inputs), chunk_size):
            end = min(start + chunk_size, len(all_inputs))
            batch_input = torch.cat(all_inputs[start:end], dim=0)
            batch_info = {
                key: torch.cat([info[key] for info in all_infos[start:end]], dim=0)
                for key in all_infos[0]
            }
            with torch.no_grad():
                chunk_output = pdfparser.model(
                    batch_input.to(pdfparser.device),
                    {k: v.to(pdfparser.device) for k, v in batch_info.items()},
                )
            all_outputs.append(chunk_output)

        batch_output = torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]

        # Split per-page results
        results = {}
        for batch_idx, page_idx in enumerate(page_indices):
            page_output = batch_output[batch_idx:batch_idx + 1]
            results[page_idx] = pdfparser.gather_output(page_output)
        return results

    def extract_all_tables_and_figures(self, pages, pdfparser, content=None):
        self.model = pdfparser
        timing = {
            'pre_filter': 0,
            'lp_forward_pass': 0,
            'page_setup': 0,
            'layout_classify': 0,
            'table_parsing': 0,
            'figure_parsing': 0,
            'active_pages': 0,
            'total_pages': len(pages),
        }
        total_start = time.perf_counter()

        # Pre-filter: skip pages with no embedded images when only extracting figures
        t0 = time.perf_counter()
        skip_pages = set()
        if content == 'figures':
            doc = fitz.open(self.pdf_file)
            for i in range(len(pages)):
                if not doc[i].get_images():
                    skip_pages.add(i)
            doc.close()
        timing['pre_filter'] = time.perf_counter() - t0

        # Collect active pages and run batched detection
        active_indices = [i for i in range(len(pages)) if i not in skip_pages]
        timing['active_pages'] = len(active_indices)

        t0 = time.perf_counter()
        if active_indices:
            active_pages = [pages[i] for i in active_indices]
            batch_results = self._detect_batched(active_pages, active_indices, pdfparser)
        else:
            batch_results = {}
        timing['lp_forward_pass'] = time.perf_counter() - t0

        ret = []
        for i in active_indices:
            t0 = time.perf_counter()
            self.set_page_num(i)
            self.img = np.asarray(pages[i])
            timing['page_setup'] += time.perf_counter() - t0

            # Set blocks from batched detection results
            t0 = time.perf_counter()
            layout_result = batch_results[i]
            self.blocks = {
                'text': lp.Layout([b for b in layout_result if b.type == 'Text']),
                'title': lp.Layout([b for b in layout_result if b.type == 'Title']),
                'list': lp.Layout([b for b in layout_result if b.type == 'List']),
                'table': lp.Layout([b for b in layout_result if b.type == 'Table']),
                'figure': lp.Layout([b for b in layout_result if b.type == 'Figure']),
            }
            timing['layout_classify'] += time.perf_counter() - t0

            if content != 'figures':
                t0 = time.perf_counter()
                table_info = self.extract_table_information()
                timing['table_parsing'] += time.perf_counter() - t0
            else:
                table_info = []
            if content != 'tables':
                t0 = time.perf_counter()
                figure_info = self.extract_figure_information()
                timing['figure_parsing'] += time.perf_counter() - t0
            else:
                figure_info = []
            if content == 'tables':
                ret += table_info
            elif content == 'figures':
                ret += figure_info
            else:
                ret += table_info
                ret += figure_info

        timing['total'] = time.perf_counter() - total_start
        self._last_timing = timing
        return ret
