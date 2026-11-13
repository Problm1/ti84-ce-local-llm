#include <ti/screen.h>
#include <ti/getcsc.h>
#include <fileioc.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint8_t** c_ptr;
static size_t* c_sz;
static int n_c = 0;

static void init_c(void) {
    c_ptr = calloc(12, sizeof(uint8_t*));
    c_sz = calloc(12, sizeof(size_t));
    char* nm = calloc(10, 1);
    for (int i = 0; i < 12; i++) {
        sprintf(nm, "MODEL%d", i);
        uint8_t h = ti_Open(nm, "r");
        if (!h) break;
        *(c_ptr + n_c) = ti_GetDataPtr(h);
        *(c_sz + n_c) = ti_GetSize(h);
        ti_Close(h);
        n_c++;
    }
    free(nm);
}

static void rd_c(size_t off, void* dst, size_t len) {
    uint8_t* o = dst;
    size_t cur = 0;
    for (int i = 0; i < n_c && len > 0; i++) {
        if (off < cur + *(c_sz + i)) {
            size_t lo = off - cur;
            size_t av = *(c_sz + i) - lo;
            size_t tk = len < av ? len : av;
            memcpy(o, *(c_ptr + i) + lo, tk);
            o += tk; off += tk; len -= tk;
        }
        cur += *(c_sz + i);
    }
}

typedef struct {
    uint32_t magic; int32_t version; int32_t dim; int32_t hidden_dim;
    int32_t n_layers; int32_t n_heads; int32_t n_kv_heads; int32_t vocab_size;
    int32_t seq_len; uint8_t shared_classifier; int32_t group_size;
} Config;

static void get_in(char* buf, int mx) {
    static const char* chs = "\0\0\0\0\0\0\0\0\0\0\"wrmh\0\0?[vqlg\0\0.zupkfc\0 ytojeb\0\0xsnida\0\0\0\0\0\0\0\0";
    int pos = 0;
    *buf = 0;
    os_PutStrFull("YOU: ");
    char* s = calloc(2, 1);
    while (1) {
        uint8_t k = os_GetCSC();
        if (k == sk_Enter && pos > 0) { os_NewLine(); break; }
        else if (k == sk_Del && pos > 0) { pos--; *(buf + pos) = 0; os_PutStrFull("\b \b"); }
        else if (k == sk_Clear) { *buf = 0; break; }
        else if (k < 56 && *(chs + k) != 0 && pos < mx - 1) {
            char c = *(chs + k);
            *(buf + pos) = c; pos++; *(buf + pos) = 0;
            *s = c; os_PutStrFull(s);
        }
    }
    free(s);
}

static char** vocab = NULL;
static float* v_sc = NULL;
static int v_sz = 512;

static int ld_tok(void) {
    uint8_t h = ti_Open("TOKEN", "r");
    if (!h)  return 0;
    uint8_t* p = ti_GetDataPtr(h);
    ti_Close(h);
    size_t off = 4;
    uint8_t *pool = (uint8_t*)0xD031F6;
    vocab = (char**)pool; pool += v_sz * sizeof(char*);
    v_sc = (float*)pool; pool += v_sz * sizeof(float);
    for (int i = 0; i < v_sz; i++) {
        memcpy(v_sc + i, p + off, 4); off += 4;
        int32_t len; memcpy(&len, p + off, 4); off += 4;
        *(vocab + i) = (char*)pool;
        memcpy(*(vocab + i), p + off, len);
        *(*(vocab + i) + len) = 0;
        pool += len + 1;
        off += len;
    }
    return 1;
}


static void mm_q8(float* xo, const int8_t* x, const float* xs, const int8_t* w, const float* ws, int n, int d, int gs) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int io = i * n;
        for (int j = 0; j <= n - gs; j += gs) {
            int32_t iv = 0;
            for (int k = 0; k < gs; k++) iv += ((int32_t)*(x + j + k)) * ((int32_t)*(w + io + j + k));
            val += ((float)iv) * *(ws + (io + j) / gs) * *(xs + j / gs);
        }
        *(xo + i) = val;
    }
}

static void quant(int8_t* qx, float* qs, const float* x, int n, int gs) {
    for (int g = 0; g < n / gs; g++) {
        float wm = 0.0f;
        for (int i = 0; i < gs; i++) {
            float v = fabsf(*(x + g * gs + i));
            if (v > wm) wm = v;
        }
        float sc = wm / 127.0f;
        *(qs + g) = sc;
        for (int i = 0; i < gs; i++) {
            float qv = (sc == 0.0f) ? 0.0f : (*(x + g * gs + i) / sc);
            *(qx + g * gs + i) = (int8_t)roundf(qv);
        }
    }
}

static void rmsn(float* o, const float* x, const float* w, int sz) {
    float ss = 0.0f;
    for (int j = 0; j < sz; j++) ss += *(x + j) * *(x + j);
    ss = 1.0f / sqrtf(ss / sz + 1e-5f);
    for (int j = 0; j < sz; j++) *(o + j) = *(w + j) * (ss * *(x + j));
}

static void sftm(float* x, int sz) {
    float mx = *x;
    for (int i = 1; i < sz; i++) if (*(x + i) > mx) mx = *(x + i);
    float sm = 0.0f;
    for (int i = 0; i < sz; i++) { *(x + i) = expf(*(x + i) - mx); sm += *(x + i); }
    for (int i = 0; i < sz; i++) *(x + i) /= sm;
}

static int enc(const char* txt, int* tok, int mx) {
    int nt = 0;
    *(tok + nt) = 2; nt++;
    char* s = calloc(8, 1);
    char* mg = calloc(128, 1);
    for (const char* p = txt; *p && nt < mx - 1; p++) {
        if (*p == 32) strcpy(s, "\xe2\x96\x81");
        else { *s = *p; *(s + 1) = 0; }
        int fnd = 1;
        for (int i = 0; i < v_sz; i++) if (strcmp(*(vocab + i), s) == 0) { fnd = i; break; }
        *(tok + nt) = fnd; nt++;
    }
    while (1) {
        float bs = -1e10f; int bi = -1, bx = -1;
        for (int i = 0; i < nt - 1; i++) {
            sprintf(mg, "%s%s", *(vocab + *(tok + i)), *(vocab + *(tok + i + 1)));
            for (int v = 0; v < v_sz; v++) {
                if (strcmp(*(vocab + v), mg) == 0 && *(v_sc + v) > bs) { bs = *(v_sc + v); bi = v; bx = i; }
            }
        }
        if (bx == -1) break;
        *(tok + bx) = bi;
        for (int i = bx + 1; i < nt - 1; i++) *(tok + i) = *(tok + i + 1);
        nt--;
    }
    free(s); free(mg);
    return nt;
}

int main(void) {
     os_ClrHome(); os_PutStrFull("LOADING TINY AI..."); os_NewLine();
    init_c();
    if (n_c == 0) { os_PutStrFull("ERR: NO MODEL"); while (!os_GetCSC()); return 1; }
    Config cfg; rd_c(0, &cfg, sizeof(Config)); v_sz = abs(cfg.vocab_size);
    if (!ld_tok()) { os_PutStrFull("ERR: NO TOKEN"); while (!os_GetCSC()); return 1; }

    int dim = cfg.dim, hdim = cfg.hidden_dim, nl = cfg.n_layers, nh = cfg.n_heads;
    int hs = dim / nh, sl = 32, gs = cfg.group_size;

    sl = 24;
    sl = 20;
    uint8_t *mp = calloc(1, 28000);
    
    float *x = (float*)mp; mp += dim * 4;
    float *xb = (float*)mp; mp += dim * 4;
    float *xb2 = (float*)mp; mp += dim * 4;
    float *hb = (float*)mp; mp += hdim * 4;
    float *hb2 = (float*)mp; mp += hdim * 4;
    float *q = (float*)mp; mp += dim * 4;
    float *kc = (float*)mp; mp += nl * sl * dim * 4;
    float *vc = (float*)0xD0EA1F;
    float *att = (float*)mp; mp += nh * sl * 4;
    float *lgt = (float*)mp; mp += v_sz * 4;
    int8_t *xq = (int8_t*)mp; mp += hdim;
    float *xs = (float*)mp; mp += (hdim / gs) * 4;
    char *ut = (char*)mp; mp += 64;
    char *pr = (char*)mp; mp += 96;
    char *ch = (char*)mp; mp += 4;
    int *tks = (int*)mp; mp += 64 * sizeof(int);
    uint8_t *wb = mp;

    os_ClrHome(); os_PutStrFull("TI-84 AI READY (CLEAR=EXIT)"); os_NewLine();

    while (1) {
        get_in(ut, 64);
        if (*ut == 0) break;
        sprintf(pr, "user: %s\nbot:", ut);
        int np = enc(pr, tks, 64);
        if (!mp) { os_PutStrFull("HEAP ERR"); while(!os_GetCSC()); return 1; } os_PutStrFull("BOT:");
        int tk = *tks;

        for (int pos = 0; pos < sl - 1; pos++) { if (pos < np - 1) os_PutStrFull(".");
            size_t off = 256, rb = nl * dim * 4;
            float *ra = (float*)wb; rd_c(off, ra, rb); off += rb;
            size_t orf = off; off += rb;
            size_t orfin = off; off += dim * 4;

            size_t eqb = (size_t)v_sz * dim, esb = (eqb / gs) * 4;
            int8_t *er = (int8_t*)(wb + rb); float *es = (float*)(wb + rb + dim);
            rd_c(off + (size_t)tk * dim, er, dim);
            rd_c(off + eqb + ((size_t)tk * dim / gs) * 4, es, (dim / gs) * 4);
            for (int i = 0; i < dim; i++) *(x + i) = *(er + i) * *(es + i / gs);
            off += eqb + esb;

            size_t qq = (size_t)nl * dim * dim, qs = (qq / gs) * 4;
            size_t owq = off; off += qq + qs; size_t owk = off; off += qq + qs;
            size_t owv = off; off += qq + qs; size_t owo = off; off += qq + qs;
            size_t fq = (size_t)nl * hdim * dim, fs = (fq / gs) * 4;
            size_t ow1 = off; off += fq + fs; size_t ow2 = off; off += fq + fs;
            size_t ow3 = off; off += fq + fs;
            size_t owc = cfg.shared_classifier ? (256 + rb * 2 + dim * 4) : off;

            for (int l = 0; l < nl; l++) {
                rmsn(xb, x, ra + l * dim, dim); quant(xq, xs, xb, dim, gs);
                size_t lqo = (size_t)l * dim * dim, lso = (lqo / gs) * 4;
                int8_t *wq = (int8_t*)(wb + rb); float *ws = (float*)(wb + rb + dim * dim);

                rd_c(owq + lqo, wq, dim * dim); rd_c(owq + qq + lso, ws, (dim * dim / gs) * 4);
                mm_q8(q, xq, xs, wq, ws, dim, dim, gs);

                float *k = kc + (l * sl + pos) * dim;
                rd_c(owk + lqo, wq, dim * dim); rd_c(owk + qq + lso, ws, (dim * dim / gs) * 4);
                mm_q8(k, xq, xs, wq, ws, dim, dim, gs);

                float *v = vc + (l * sl + pos) * dim;
                rd_c(owv + lqo, wq, dim * dim); rd_c(owv + qq + lso, ws, (dim * dim / gs) * 4);
                mm_q8(v, xq, xs, wq, ws, dim, dim, gs);

                for (int i = 0; i < dim; i += 2) {
                    float fr = 1.0f / powf(10000.0f, (i % hs) / (float)hs), vl = pos * fr;
                    float fc = cosf(vl), fi = sinf(vl), v0 = *(q + i), v1 = *(q + i + 1);
                    *(q + i) = v0 * fc - v1 * fi; *(q + i + 1) = v0 * fi + v1 * fc;
                    v0 = *(k + i); v1 = *(k + i + 1);
                    *(k + i) = v0 * fc - v1 * fi; *(k + i + 1) = v0 * fi + v1 * fc;
                }

                for (int h = 0; h < nh; h++) {
                    float *qh = q + h * hs, *ah = att + h * sl;
                    for (int t = 0; t <= pos; t++) {
                        float *kh = kc + (l * sl + t) * dim + h * hs, sc = 0.0f;
                        for (int i = 0; i < hs; i++) sc += *(qh + i) * *(kh + i);
                        *(ah + t) = sc / sqrtf((float)hs);
                    }
                    sftm(ah, pos + 1);
                    float *xbh = xb + h * hs; memset(xbh, 0, hs * 4);
                    for (int t = 0; t <= pos; t++) {
                        float *vh = vc + (l * sl + t) * dim + h * hs, a = *(ah + t);
                        for (int i = 0; i < hs; i++) *(xbh + i) += a * *(vh + i);
                    }
                }

                quant(xq, xs, xb, dim, gs);
                rd_c(owo + lqo, wq, dim * dim); rd_c(owo + qq + lso, ws, (dim * dim / gs) * 4);
                mm_q8(xb2, xq, xs, wq, ws, dim, dim, gs);
                for (int i = 0; i < dim; i++) *(x + i) += *(xb2 + i);

                float *rf = (float*)(wb + rb);
                rd_c(orf + l * dim * 4, rf, dim * 4);
                rmsn(xb, x, rf, dim); quant(xq, xs, xb, dim, gs);

                size_t lfo = (size_t)l * hdim * dim, lfso = (lfo / gs) * 4;
                int8_t *wf = (int8_t*)(wb + rb); float *wfs = (float*)(wb + rb + hdim * dim);

                rd_c(ow1 + lfo, wf, hdim * dim); rd_c(ow1 + fq + lfso, wfs, (hdim * dim / gs) * 4);
                mm_q8(hb, xq, xs, wf, wfs, dim, hdim, gs);

                rd_c(ow3 + lfo, wf, hdim * dim); rd_c(ow3 + fq + lfso, wfs, (hdim * dim / gs) * 4);
                mm_q8(hb2, xq, xs, wf, wfs, dim, hdim, gs);

                for (int i = 0; i < hdim; i++) {
                    float vl = *(hb + i); vl *= (1.0f / (1.0f + expf(-vl))); *(hb + i) = vl * *(hb2 + i);
                }

                quant(xq, xs, hb, hdim, gs);
                rd_c(ow2 + lfo, wf, hdim * dim); rd_c(ow2 + fq + lfso, wfs, (hdim * dim / gs) * 4);
                mm_q8(xb, xq, xs, wf, wfs, hdim, dim, gs);
                for (int i = 0; i < dim; i++) *(x + i) += *(xb + i);
            }

            float *rfin = (float*)(wb + rb);
            rd_c(orfin, rfin, dim * 4);
            rmsn(x, x, rfin, dim); quant(xq, xs, x, dim, gs);

            if (pos < np - 1) { tk = *(tks + pos + 1); continue; } int8_t *wc = (int8_t*)(wb + rb); float *wcs = (float*)(wb + rb + dim);
            for (int r = 0; r < v_sz; r++) { rd_c(owc + (size_t)r * dim, wc, dim); rd_c(owc + (size_t)v_sz * dim + ((size_t)r * dim / gs) * 4, wcs, (dim / gs) * 4); mm_q8(lgt + r, xq, xs, wc, wcs, dim, 1, gs); }
            

            int nt = 0;
            if (pos < np - 1) {
                nt = *(tks + pos + 1);
            } else {
                float bst = *lgt;
                for (int i = 1; i < v_sz; i++) if (*(lgt + i) > bst) { bst = *(lgt + i); nt = i; }
                if (nt == 3 || nt == 0) break;
                char *pc = *(vocab + nt);
                if (strchr(pc, 10)) break;
                for (int c = 0; *(pc + c); ) {
                    if ((uint8_t)*(pc + c) == 0xe2 && (uint8_t)*(pc + c + 1) == 0x96 && (uint8_t)*(pc + c + 2) == 0x81) {
                        os_PutStrFull(" "); c += 3;
                    } else {
                        *ch = *(pc + c); os_PutStrFull(ch); c++;
                    }
                }
            }
            tk = nt;
        }
        os_NewLine();
    }
     return 0;
}
