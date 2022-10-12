import pynini
from collections import defaultdict


class VocabFSA:

    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

        self.start_state = 0
        self.state_dict = defaultdict(lambda: defaultdict(None))
        self.cur_state = self.start_state
        self.last_token_id = None

        self.assemble_fsa()

    def encode(self, s):
        """ Shift all token ids up 1 to avoid using the token 0, which is
            interpreted as epsilon in pynini """
        return tuple(tok_id + 1 for tok_id in self.tokenizer.encode(s))

    def decode(self, tokens):
        """ Shift all token ids back down 1 to their original id """
        return tuple(self.tokenizer.decode(tok - 1) for tok in tokens)

    def batch_encode(self, strings):
        """ Iterate over token strings, returning a list of tuples of token ids """
        ret = []
        for s in strings:
            ret.append(self.encode(s))
        return ret

    @staticmethod
    def get_token_variants_cap(term):
        """ Retrieve tokens for possible variations on a vocabulary item, with
            different capitalization schemas. No surrounding whitespace.

            :param tokenizer: a HuggingFace tokenizer object that converts strings to token ids
            :type tokenizer: AutoTokenizer
            :param term: a vocabulary item
            :type term: str
            :return: set of tokens
            :rtype: set{str}
        """
        term = term.strip()
        initial_cap = term[0].upper() + term[1:]
        all_caps = term.upper()

        return {initial_cap, all_caps}

    @staticmethod
    def get_token_variants_ws(term):
        """ Retrieve tokens for possible variations on a vocabulary item, with an
            initial white space.

            :param tokenizer: a HuggingFace tokenizer object that converts strings to token ids
            :type tokenizer: AutoTokenizer
            :param term: a vocabulary item
            :type term: str
            :return: list of lists of token ids
            :rtype: list[list[int]]
        """
        term = term.strip()
        initial_cap = term[0].upper() + term[1:]
        all_caps = term.upper()
        i_space = ' ' + term
        i_space_cap = ' ' + initial_cap
        i_space_all_caps = ' ' + all_caps

        return {i_space, i_space_cap, i_space_all_caps}

    def get_variant_token_ids(self):
        cap_token_ids = set()
        ws_token_ids = set()

        for term in self.vocab:
            cap_tokens = self.get_token_variants_cap(term)
            ws_tokens = self.get_token_variants_ws(term)

            for tok_ids in self.batch_encode(cap_tokens):
                cap_token_ids.add(tok_ids)

            for tok_ids in self.batch_encode(ws_tokens):
                ws_token_ids.add(tok_ids)

        return cap_token_ids, ws_token_ids

    def get_punc_ids(self):
        punc_ids = set()

        punct = ".,?!;:'\"”)/…"
        for p in punct:
            punc_ids.add((self.encode(p)[0],))
            for q in punct:
                punc_ids.add((self.encode(p + q)[0],))
                for r in punct:
                    punc_ids.add((self.encode(p + q + r)[0],))

        return punc_ids

    def get_ws_ids(self):
        ws_ids = set()

        ws_chars = " \t\n"
        for tok_ids in self.batch_encode(ws_chars):
            ws_ids.add(tok_ids)

        ws_ids.add((self.tokenizer.eos_token_id,))

        return ws_ids

    def build_pynini_fsa(self, token_ids):
        fsts = []
        for tok_chain in token_ids:
            tok_id_str = ''.join([chr(tok_id) for tok_id in tok_chain])
            fsts.append(pynini.accep(tok_id_str, token_type="utf8"))

        return pynini.union(*fsts).optimize()

    def build_sub_fsas(self):
        cap_token_ids, ws_token_ids = self.get_variant_token_ids()
        punc_ids = self.get_punc_ids()
        ws_ids = self.get_ws_ids()

        cap_token_fst = self.build_pynini_fsa(cap_token_ids)
        ws_token_fst = self.build_pynini_fsa(ws_token_ids)
        punc_fst = self.build_pynini_fsa(punc_ids)
        ws_fst = self.build_pynini_fsa(ws_ids)

        return cap_token_fst, ws_token_fst, punc_fst, ws_fst

    @staticmethod
    def is_final(fst, state):
        """ Given an fst (pynini.Fst) and a state (int), return True if the
            state is final, False if it isn't. """
        return float(fst.final(state)) == 0

    def get_final(self, fst):
        """ Iterate over states in an FST and return the final state as an int. """
        for state in fst.states():
            if self.is_final(fst, state):
                return state
        return None

    def read_fst(self, fst, offset=0, start_state=None, final_state=None):
        for orig_state in fst.states():
            if fst.start() == orig_state and start_state is not None:
                new_state = start_state
            else:
                new_state = orig_state + offset

            for arc in fst.arcs(orig_state):
                token = arc.ilabel - 1  # Shift tokens back down so they align with the actual model
                if self.is_final(fst, arc.nextstate) and final_state is not None:
                    self.state_dict[new_state][token] = final_state
                else:
                    self.state_dict[new_state][token] = arc.nextstate + offset

        new_offset = offset + len([state for state in fst.states()]) - 1

        if final_state is not None:
            new_offset -= 1
        else:
            final_state = self.get_final(fst) + offset

        return new_offset, final_state

    def assemble_fsa(self):
        cap_token_fst, ws_token_fst, punc_fst, ws_fst = self.build_sub_fsas()
        self.start_state = cap_token_fst.start()

        offset, cap_final_state = self.read_fst(cap_token_fst)
        offset, ws_final_state = self.read_fst(ws_token_fst, offset, start_state=cap_final_state)
        offset, punc_final_state = self.read_fst(punc_fst, offset, start_state=cap_final_state)
        offset, _ = self.read_fst(ws_fst, offset, start_state=cap_final_state, final_state=self.start_state)
        offset, _ = self.read_fst(ws_token_fst, offset, start_state=ws_final_state, final_state=ws_final_state)
        offset, _ = self.read_fst(punc_fst, offset, start_state=ws_final_state, final_state=punc_final_state)
        offset, _ = self.read_fst(ws_fst, offset, start_state=ws_final_state, final_state=self.start_state)
        offset, _ = self.read_fst(ws_token_fst, offset, start_state=punc_final_state, final_state=ws_final_state)
        offset, _ = self.read_fst(ws_fst, offset, start_state=punc_final_state, final_state=self.start_state)
        offset, _ = self.read_fst(ws_fst, offset, start_state=self.start_state, final_state=self.start_state)

    def advance(self, token_id):
        """ Given the following token_id, update the current state. """
        assert self.cur_state in self.state_dict, f"{self.cur_state} is not a valid state."
        assert token_id in self.state_dict[self.cur_state], f"{token} is not a valid token for the state {self.cur_state}"

        self.last_token_id = token_id
        self.cur_state = self.state_dict[self.cur_state][self.last_token_id]

    def next_tokens(self):
        """ Return a list of the next possible token ids given the current state. """
        return list(self.state_dict[self.cur_state].keys())

    def __repr__(self):
        return "cur_state: {}\nlast_token: {}".format(self.cur_state, self.last_token_id)

