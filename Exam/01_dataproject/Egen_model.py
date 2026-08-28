import numpy as np

def setup():
    """ Definerer modellens parametre """

    par = dict() 

    # antal personer og alder
    par['N'] = 50_000

    # uddannelse: (kort, mellem, lang)
    par['p_e'] = (0.40, 0.35, 0.25)      # sandsynlighed for hver type. Skal bruges længere nede til at trække uddannelsestype for hver person.
    par['S_e'] = (1, 3, 5)               # år under uddannelse
    par['h_e0'] = (1.00, 1.20, 1.55)     # startniveau af humankapital
    par['delta_e'] = (0.010, 0.020, 0.030)  # vækstrate i humankapital

    # arbejdsmarked
    par['lambda_'] = 0.60   # sandsynlighed for at finde job
    par['sigma'] = 0.05     # sandsynlighed for at miste job
    par['delta'] = 0.06     # depreciering af humankapital som ledig
    par['sigma_psi'] = 0.10 # std. af shock til humankapital

    # indkomst
    par['y_SU'] = 0.45   # SU
    par['rho'] = 0.60    # kompensationsgrad som ledig
    par['y_floor'] = 0.35  # gulv for indkomst

    # antal år der simuleres (alder 18 til 64, da man går på pension som 65-årig)
    par['T'] = 65 - 18
    
    return par


def allocate(par):
    """ opretter tomme arrays til at gemme simuleringen i """

    N = par['N']
    T = par['T']

    sim = dict()

    sim['education'] = np.zeros(N, dtype=int)         # 0=kort, 1=mellem, 2=lang (fast for hver person). Dette er strengt taget ikke nødvendigt da vi senere overskriver hele arrayet. Men det holder stringens i forhold til de andre arrays, som skal fyldes med data for hvert år.
    sim['human_capital'] = np.zeros((N, T))           # humankapital, person x år
    sim['income'] = np.zeros((N, T))                  # indkomst, person x år
    sim['employed'] = np.zeros((N, T), dtype=bool)    # True = i job, False = ledig/under uddannelse

    # Alle ovenstående arrays er printet i notebooken Code_Results

    return sim

def draw_education(par, sim, rng):
    """ trækker uddannelsestype for hver person """

    sim['education'] = rng.choice(3, size=par['N'], p=par['p_e'])  # Tager tilfældige tal mellem 1-3 (fordi den får input 3) med sandsynlighederne p_e. Dette er en vektor med N elementer, hvor hvert element er 0, 1 eller 2, der repræsenterer uddannelsestypen for hver person.

def simulate(par, sim, rng, sigma_xi=0.0, verbose=True):
    """ simulérer modellen for alle N personer over alle T år

    Udover uddannelsesdelen (som vi lavede ovenfor) tilføjer denne funktion:
      - arbejdsmarkedet: man er ledig eller i job, og skifter mellem de to
        med sandsynlighederne lambda (finder job) og sigma (mister job)
      - humankapital: vokser mens man er i job, falder mens man er ledig,
        og bliver ramt af et tilfældigt chok (psi) hvert år
      - indkomst: SU under uddannelse, humankapital hvis i job, en andel
        (rho) af sidste løn hvis ledig, og et gulv hvis man aldrig har haft job

    Ekstra argumenter (bruges først senere i opgaven, derfor default-værdier):
        sigma_xi: ekstra lønrisiko for folk i job, kun brugt i opgave 2.5.
                  Sat til 0 som standard, så den ikke ændrer noget i baseline-modellen.
        verbose:  hvis True printes antal personer i uddannelse for de første år (opgave 2.1).
                  Sættes til False når vi skal køre modellen mange gange, fx i opgave 2.4,
                  for ikke at drukne i print-output.
    """

    N = par['N']
    T = par['T']

    # antal år hver person skal bruge på uddannelse, samt det humankapital-niveau
    # og den vækstrate de starter arbejdslivet med - alle lavet med fancy indexing,
    # ligesom years_in_education herunder.
    S_e = np.array(par['S_e'])
    h_e0 = np.array(par['h_e0'])
    delta_e = np.array(par['delta_e'])

    years_in_education = S_e[sim['education']]     # antal år i uddannelse, ét tal per person
    start_humankapital = h_e0[sim['education']]     # humankapital de starter med som færdiguddannede
    egen_vaekstrate = delta_e[sim['education']]     # deres egen vækstrate i humankapital, hvis i job

    # "tilstands"-variable der bæres med fra ét år til det næste. Alle starter ved
    # 0/False - det bliver først sat til noget rigtigt, når personen bliver færdig
    # med sin uddannelse og kommer ud på arbejdsmarkedet, se punkt i. nedenfor.
    h = np.zeros(N)                              # humankapital lige nu
    employed = np.zeros(N, dtype=bool)            # er personen i job lige nu?
    sidste_jobloen = np.full(N, par['y_floor'])   # løn fra sidste job (bruges hvis man senere bliver ledig)
    har_vaeret_i_job = np.zeros(N, dtype=bool)    # har personen nogensinde haft et job?

    for t in range(T):

        # er personen stadig under uddannelse i år t, eller er hun/han "aktiv" på
        # arbejdsmarkedet? "entering" er sand i det ene år hvor personen lige er
        # blevet færdig med sin uddannelse.
        in_education = t < years_in_education
        active = ~in_education
        entering = (t == years_in_education)

        # i. de der lige er kommet ud på arbejdsmarkedet starter som ledige, med
        #    deres uddannelses-specifikke startniveau af humankapital
        h = np.where(entering, start_humankapital, h)
        employed = np.where(entering, False, employed)

        # ii. opgave 2.5: hvis sigma_xi > 0 trækkes et ekstra (mean-one lognormalt)
        #     lønchok til dem der er i job. Ellers sættes xi = 1, dvs. ingen effekt.
        if sigma_xi > 0:
            xi = rng.lognormal(-0.5 * sigma_xi**2, sigma_xi, size=N)
        else:
            xi = 1.0

        # iii. indkomst i år t. Vi bruger tre np.where "inde i hinanden", ét for
        #      hver case i opgaven:
        #        - under uddannelse         -> SU
        #        - i job                    -> humankapital (x en evt. ekstra lønrisiko)
        #        - ledig, har haft job før  -> en andel (rho) af lønnen fra sidste job
        #        - ledig, aldrig haft job   -> gulvet y_floor
        income = np.where(
            in_education, par['y_SU'],
            np.where(
                employed, h * xi,
                np.where(har_vaeret_i_job, par['rho'] * sidste_jobloen, par['y_floor'])
            )
        )

        sim['income'][:, t] = income
        sim['employed'][:, t] = employed & active   # kun "i job" hvis man rent faktisk er på arbejdsmarkedet
        sim['human_capital'][:, t] = h               # 0 for dem der stadig er under uddannelse - bruges ikke der

        # iv. opdaterer "hukommelsen" om jobhistorik, kun for dem der er aktive og i job lige nu
        i_job_nu = active & employed
        sidste_jobloen = np.where(i_job_nu, income, sidste_jobloen)
        har_vaeret_i_job = har_vaeret_i_job | i_job_nu

        # v. trækker de tilfældige tal der bestemmer, hvad der sker frem til næste år
        psi = rng.lognormal(-0.5 * par['sigma_psi']**2, par['sigma_psi'], size=N)  # chok til humankapital
        u = rng.random(N)                                                          # bruges til jobskifte

        # vi. humankapital næste år: vokser med ens egen vækstrate hvis i job,
        #     falder med delta (depreciering) hvis ledig - begge dele ganget med choket psi
        h_naeste = np.where(employed, h * (1 + egen_vaekstrate) * psi, h * (1 - par['delta']) * psi)
        h = np.where(active, h_naeste, h)   # opdater kun for dem der reelt er kommet ud på arbejdsmarkedet

        # vii. jobstatus næste år: i job -> mister det med sandsynlighed sigma
        #                          ledig -> finder et job med sandsynlighed lambda
        employed_naeste = np.where(employed, u > par['sigma'], u < par['lambda_'])
        employed = np.where(active, employed_naeste, employed)

        if verbose and t < 6:  # kun print for de første par år, ellers drukner du i output
            print(f't={t}: {in_education.sum()} personer er stadig under uddannelse')


def gini(y):
    """ beregner gini-koefficienten for en 1d-vektor af indkomster y (opgave 2.3)

    Vi bruger den "sorterede" formel for gini-koefficienten:

        G = (2 * sum(rank_i * y_i) - (n+1) * sum(y_i)) / (n * sum(y_i))

    hvor y først sorteres i stigende orden, og rank_i = 1, 2, ..., n er
    indkomstens placering i den sorterede liste (1 = den laveste indkomst).
    Formlen er testet i notebooken mod tilfælde hvor vi kender facit
    (uniform og lognormal fordeling), se opgave 2.3.
    """

    y_sorteret = np.sort(y)          # sorterer indkomsterne fra lav til høj
    n = len(y_sorteret)
    rank = np.arange(1, n + 1)       # 1, 2, ..., n - pladsen i den sorterede liste

    G = (2 * np.sum(rank * y_sorteret) - (n + 1) * np.sum(y_sorteret)) / (n * np.sum(y_sorteret))

    return G


def gini_lorenz(y):
    """ beregner gini-koefficienten via Lorenz-kurven (opgave 2.3)

    Følger figuren direkte:
      1) byg den empiriske Lorenz-kurve (kumuleret indkomstandel som funktion
         af kumuleret befolkningsandel)
      2) integrer arealet B under kurven med trapez-metoden
      3) brug at A + B = 1/2 (trekanten under 45-graders linjen), så
         G = A / (A + B) = 2*A = 1 - 2*B
    """

    y_sorteret = np.sort(y)
    n = len(y_sorteret)

    x = np.arange(0, n + 1) / n                                          # andel af befolkningen: 0, 1/n, ..., n/n
    L = np.concatenate(([0], np.cumsum(y_sorteret))) / y_sorteret.sum()   # andel af indkomst: L(0)=0, L(x_i) = kumuleret andel

    # trapez-metoden: arealet under en stykvis lineær kurve er summen af
    # trapez-arealer mellem hvert par af nabopunkter
    B = np.sum((x[1:] - x[:-1]) * (L[1:] + L[:-1]) / 2)

    # Beregner gini-koefficienten ud fra arealet B under Lorenz-kurven. Arealet A over Lorenz-kurven er 0.5 - B, og gini-koefficienten er A / (A + B) = 2*A = 1 - 2*B.
    G = 1 - 2 * B

    return G
