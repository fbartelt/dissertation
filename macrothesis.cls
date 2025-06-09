\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{macrothesis}[2024/11/01 Macro Thesis Template]

\RequirePackage{ifthen}
\RequirePackage{etoolbox} % For boolean flags
% ---------------------------------------------------------------------------- %
%                              DEFAULT VALUES
% ---------------------------------------------------------------------------- %
\newcommand{\@useglossaries}{true}
\newcommand{\@usenatbib}{true}
\newcommand{\@usehyperref}{true}
\newcommand{\@usecleverref}{true}
\newcommand{\@mainlanguage}{english}
\newcommand{\@secondlanguage}{none}
\newcommand{\@texttype}{thesis}
\newcommand{\@masterdoc}{master}
\newcommand{\@universityName}{Universidade Federal de Minas Gerais}
\newcommand{\@programName}{Graduate Program in Electrical Engineering}
\newcommand{\@researchGroupName}{MACRO Research Group - Mechatronics, Control, and Robotics}
\newcommand{\@coursename}{Electrical Engineering}
\newcommand{\@locationName}{Belo Horizonte, Brazil}
\newcommand{\@notationpage}{}
\newcommand{\@ufmgformat}{false}

\newcommand{\supportedlangs}{english, american, USenglish, canadian, british, 
australian, newzealand, UKenglish, brazil, portuguese, german, ngerman, spanish, 
french}
\newcommand{\titlelang}{Thesis}
\newcommand{\typelang}{Master}
\newcommand{\advisorlang}{Advisor}
\newcommand{\coadvisorlang}{Co-Advisor}
\newcommand{\submissiontext}{{\typelang} submitted to the {\@programName} of {\@universityName}, in partial fulfillment of the requirements for the degree of {\titlelang} in {\@coursename}}

\newbool{hasfichacatalografica}
\boolfalse{hasfichacatalografica} % Default: false
\newcommand{\@fichapath}{} % Stores the path
\newbool{hasatadefesa}
\boolfalse{hasatadefesa} % Default: false
\newcommand{\@atapath}{} % Stores the path
% ---------------------------------------------------------------------------- %
%                              IMPORT OPTIONS
% ---------------------------------------------------------------------------- %
\DeclareOption{noglossaries}{%
    \renewcommand{\@useglossaries}{false}%
}
\DeclareOption{nonatbib}{%
    \renewcommand{\@usenatbib}{false}%
}
\DeclareOption{nohyperref}{%
    \renewcommand{\@usehyperref}{false}%
}
\DeclareOption{nocleverref}{%
    \renewcommand{\@usecleverref}{false}%
}
\DeclareOption{final}{%
    \renewcommand{\@ufmgformat}{true}%
}
% ---------------------------------------------------------------------------- %
%                              LANGUAGE OPTIONS
% ---------------------------------------------------------------------------- %
\DeclareOption{firstlang=english}{%
  \renewcommand{\@mainlanguage}{english}%
}
\DeclareOption{firstlang=american}{%
  \renewcommand{\@mainlanguage}{american}%
}
\DeclareOption{firstlang=USenglish}{%
  \renewcommand{\@mainlanguage}{USenglish}%
}
\DeclareOption{firstlang=canadian}{%
  \renewcommand{\@mainlanguage}{canadian}%
}
\DeclareOption{firstlang=british}{%
  \renewcommand{\@mainlanguage}{british}%
}
\DeclareOption{firstlang=australian}{%
  \renewcommand{\@mainlanguage}{australian}%
}
\DeclareOption{firstlang=newzealand}{%
  \renewcommand{\@mainlanguage}{newzealand}%
}
\DeclareOption{firstlang=UKenglish}{%
  \renewcommand{\@mainlanguage}{UKenglish}%
}
\DeclareOption{firstlang=brazil}{%
  \renewcommand{\@mainlanguage}{brazil}%
}
\DeclareOption{firstlang=portuguese}{%
  \renewcommand{\@mainlanguage}{portuguese}%
}
\DeclareOption{firstlang=german}{%
  \renewcommand{\@mainlanguage}{german}%
}
\DeclareOption{firstlang=ngerman}{%
  \renewcommand{\@mainlanguage}{ngerman}%
}
\DeclareOption{firstlang=spanish}{%
  \renewcommand{\@mainlanguage}{spanish}%
}
\DeclareOption{firstlang=french}{%
  \renewcommand{\@mainlanguage}{french}%
}

% Define options for secondlang
\DeclareOption{secondlang=english}{%
  \renewcommand{\@secondlanguage}{english}%
}
\DeclareOption{secondlang=american}{%
  \renewcommand{\@secondlanguage}{american}%
}
\DeclareOption{secondlang=USenglish}{%
  \renewcommand{\@secondlanguage}{USenglish}%
}
\DeclareOption{secondlang=canadian}{%
  \renewcommand{\@secondlanguage}{canadian}%
}
\DeclareOption{secondlang=british}{%
  \renewcommand{\@secondlanguage}{british}%
}
\DeclareOption{secondlang=australian}{%
  \renewcommand{\@secondlanguage}{australian}%
}
\DeclareOption{secondlang=newzealand}{%
  \renewcommand{\@secondlanguage}{newzealand}%
}
\DeclareOption{secondlang=UKenglish}{%
  \renewcommand{\@secondlanguage}{UKenglish}%
}
\DeclareOption{secondlang=brazil}{%
  \renewcommand{\@secondlanguage}{brazil}%
}
\DeclareOption{secondlang=portuguese}{%
  \renewcommand{\@secondlanguage}{portuguese}%
}
\DeclareOption{secondlang=german}{%
  \renewcommand{\@secondlanguage}{german}%
}
\DeclareOption{secondlang=ngerman}{%
  \renewcommand{\@secondlanguage}{ngerman}%
}
\DeclareOption{secondlang=spanish}{%
  \renewcommand{\@secondlanguage}{spanish}%
}
\DeclareOption{secondlang=french}{%
  \renewcommand{\@secondlanguage}{french}%
}
% ---------------------------------------------------------------------------- %
%                              TEXT OPTIONS
% ---------------------------------------------------------------------------- %
\DeclareOption{thesis}{%
    \renewcommand{\@texttype}{thesis}%
}
\DeclareOption{dissertation}{%
    \renewcommand{\@texttype}{dissertation}%
}
\DeclareOption{master}{%
    \renewcommand{\@masterdoc}{master}%
}
\DeclareOption{doctor}{%
    \renewcommand{\@masterdoc}{doctor}%
}
% Process options
\ProcessOptions\relax

% ---------------------------------------------------------------------------- %
%                              REQUIRED PACKAGES
% ---------------------------------------------------------------------------- %
\LoadClass[12pt,a4paper,oneside,openany]{book}
% Load additional packages
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

\RequirePackage{pdfpages} % For PDF insertion -- Ficha Catalografica + Ata de defesa

% Load Babel with secondary language only
% Load languages using babel
\ifthenelse{\equal{\@secondlanguage}{none}}{%
  \RequirePackage[main=\@mainlanguage]{babel}
  \renewcommand{\@secondlanguage}{\@mainlanguage} % avoids errors when switching
}{%
  \RequirePackage[main=\@mainlanguage,\@secondlanguage]{babel}
}
\RequirePackage{graphicx} % Required for including images
\RequirePackage[a4paper]{geometry} % Required for adjusting page dimensions and margins
\geometry{verbose,tmargin=3cm,bmargin=2cm,lmargin=3cm,rmargin=2cm,headheight=15pt}
\RequirePackage{setspace} % Required for changing line spacing
\RequirePackage{microtype} % Improves the spacing between words and letters
\onehalfspacing

% Other packages
\RequirePackage{fancyhdr} % Required for custom headers -- Whole style of template
\RequirePackage{xcolor} % Required for specifying colors by name
\RequirePackage{lmodern} % Latin Modern font
\RequirePackage[lmodern]{quotchap} % used to customize the appearance of chapter titles, often adding a quotation or some style modification
% \RequirePackage{algorithmic}
\RequirePackage[nottoc]{tocbibind} % Add bibliography/index/contents to Table of Contents, not including ToC itself
\RequirePackage{import} % Used to import files from other directories
\RequirePackage{caption} % Required for customizing captions
\RequirePackage{subcaption} % Required for subfigures
% \RequirePackage{ae}
\RequirePackage{amsmath,amssymb,amsthm,amsfonts,thmtools,mathrsfs} % Required for math
\RequirePackage{scrbase} % Required for handling language-dependent names of sections/document elements
% \RequirePackage{dcolumn}
\RequirePackage{enumitem} % Required for customizing lists
% ---------------------------------------------------------------------------- %
%                              OPTIONAL PACKAGES
% ---------------------------------------------------------------------------- %
\ifthenelse{\equal{\@usenatbib}{true}}{%
    \RequirePackage[authoryear,round]{natbib}
}{%
    \PackageInfo{macrothesis}{natbib package not loaded.}%
}
\ifthenelse{\equal{\@usehyperref}{true}}{%
    \RequirePackage[pdfusetitle,
    bookmarks=true,bookmarksnumbered=false,bookmarksopen=false,
    breaklinks=true,backref=false,colorlinks=true,allcolors=teal]
    {hyperref}
}{%
    \PackageInfo{macrothesis}{hyperref package not loaded.}%
}
\ifthenelse{\equal{\@usecleverref}{true}}{%
    \RequirePackage[capitalize,nameinlink]{cleveref}
}{%
    \PackageInfo{macrothesis}{cleverref package not loaded.}%
}
% ---------------------------------------------------------------------------- %
%                              CLASS FEATURES
% ---------------------------------------------------------------------------- %
% Custom theorem environments
\declaretheoremstyle[qed=\qedsymbol]{defnstyle}
\declaretheorem[name=Theorem,numberwithin=chapter]{theorem}
\declaretheorem[name=Proposition,numberwithin=chapter,sibling=theorem]{proposition}
\declaretheorem[name=Lemma,numberwithin=chapter,sibling=theorem]{lemma}
\declaretheorem[name=Corollary,numberwithin=chapter,sibling=theorem]{corollary}
\declaretheorem[name=Example,numberwithin=chapter,sibling=theorem,style=defnstyle]{example}
\declaretheorem[name=Remark,numberwithin=chapter,sibling=theorem,style=defnstyle]{remark}
\declaretheorem[name=Definition,numberwithin=chapter,sibling=theorem,style=defnstyle]{definition}
\declaretheorem[name=Assumption,numberwithin=chapter,sibling=theorem,style=defnstyle]{assumption}
\declaretheorem[name=Claim,numberwithin=chapter,sibling=theorem,style=defnstyle]{claim}
\declaretheorem[name=Conjecture,numberwithin=chapter,sibling=theorem,style=defnstyle]{conjecture}
\declaretheorem[name=Fact,numberwithin=chapter,sibling=theorem,style=defnstyle]{fact}
\declaretheorem[name=Problem,numberwithin=chapter,sibling=theorem,style=defnstyle]{problem}
\declaretheorem[name=Solution,numberwithin=chapter,sibling=theorem,style=defnstyle]{solution}
\declaretheorem[name=Criterion,numberwithin=chapter,sibling=theorem,style=defnstyle]{criterion}

% Custom lists
\newlist{property}{enumerate}{1}
\newlist{feature}{enumerate}{1}
\setlist[property]{label=\roman{propertyi}, ref=\thedefinition.\roman{propertyi},noitemsep}
\setlist[feature]{label=(\alph*), ref=(\alph*),noitemsep}

% Cleverref settings (if loaded)
\ifthenelse{\equal{\@usecleverref}{true}}{%
  \Crefname{theorem}{Theorem}{Theorems}
  \Crefname{proposition}{Proposition}{Propositions}
  \Crefname{lemma}{Lemma}{Lemmas}
  \Crefname{definition}{Definition}{Definitions}
  \Crefname{corollary}{Corollary}{Corollaries}
  \Crefname{listprop}{Property}{Properties}
  \Crefname{example}{Example}{Examples}
  \Crefname{remark}{Remark}{Remarks}
  \Crefname{assumption}{Assumption}{Assumptions}
  \Crefname{claim}{Claim}{Claims}
  \Crefname{conjecture}{Conjecture}{Conjectures}
  \Crefname{fact}{Fact}{Facts}
  \Crefname{problem}{Problem}{Problems}
  \Crefname{solution}{Solution}{Solutions}
  \Crefname{criterion}{Criterion}{Criteria}

  \Crefname{listprop}{Property}{Properties}
  \addtotheorempostheadhook[definition]{\crefalias{propertyi}{listprop}}
  \crefname{featurei}{Feature}{Features}
  \Crefname{figure}{Figure}{Figures}
}{%
}
% Change footnote numbering to symbols in minipage
\renewcommand{\thempfootnote}{\,\fnsymbol{mpfootnote}}
\fancypagestyle{thesisstyle}{%
    \fancyhf{} % Clear all headers/footers
    \fancyhead[R]{\thepage} % Page number always on right
    \fancyhead[L]{\small\itshape\rightmark} % Chapter title always on left
    \renewcommand{\headrulewidth}{0pt} % No header rule
    \renewcommand{\footrulewidth}{0pt} % No footer rule
  }
% ---------------------------------------------------------------------------- %
%                WORKAROUND TO REMOVE PAGE NUMBERS IN PREAMBLE
% ---------------------------------------------------------------------------- %
% All of these are for final version only (option 'final')
\newcommand{\removestyle}{
  % Workaround to avoid page numbers on the first pages of chapters (preamble)
  \fancypagestyle{plain}{%
    \fancyhf{} % No headers or footers
    \renewcommand{\headrulewidth}{0pt} % No header rule
    \renewcommand{\footrulewidth}{0pt} % No footer rule
  }
}

\newcommand{\setfancyheader}{
  % Part of the workaround -- sets page numbers to top right corner for chapters
  \fancypagestyle{plain}{%
    \fancyhf{} % Clear all headers/footers
    \fancyhead[R]{\thepage} % Page number always on right
    \fancyhead[L]{\small\itshape\rightmark} % Chapter title always on left
    \renewcommand{\headrulewidth}{0pt} % No header rule
    \renewcommand{\footrulewidth}{0pt} % No footer rule
  }
  \pagestyle{plain} % Set the page style to the modified plain
}
% ---------------------------------------------------------------------------- %
%                              USER COMMANDS
% ---------------------------------------------------------------------------- %

% Command to store thesis title
% \newcommand{\thesistitle}[1]{\def\@thesisTitle{#1}}
\def\@thesisTitle{\@title}
% Command to store author name
% \newcommand{\authorname}[1]{\def\@thesisAuthor{#1}}
\def\@thesisAuthor{\@author}
% Command to store advisor name
\newcommand{\advisorname}[1]{\def\@thesisAdvisor{#1}}
% Command to store co-advisor names (up to 2)
\newcommand{\coadvisorname}[2][]{%
    \ifx&#1& % Check if the optional argument is empty
        \def\@thesisCoadvisorB{} % If empty, set second coadvisor to empty
    \else
        \def\@thesisCoadvisorB{#1} % Set to the provided optional argument
    \fi

    \ifx&#2& % Check if the mandatory argument is empty
        \def\@thesisCoadvisorA{} % If empty, set first coadvisor to empty
    \else
        \def\@thesisCoadvisorA{#2} % Set to the provided mandatory argument
    \fi
}
% Command to store the program name
\newcommand{\programname}[1]{\renewcommand{\@programName}{#1}}
% Command to store university name
\newcommand{\universityname}[1]{\renewcommand{\@universityName}{#1}}
% Command to store location (city/state)
\newcommand{\locationname}[1]{\renewcommand{\@locationName}{#1}}
% Command to store research group name
\newcommand{\researchgroupname}[1]{\renewcommand{\@researchGroupName}{#1}}
% Command to store dedication text
\newcommand{\dedication}[1]{\def\@dedicationContent{#1}}
% Command to store acknowledgments text
\newcommand{\acknowledgments}[2][\acknowledgmentsname]{%
    \def\@acknowledgmentsTitle{#1} % Set title
    \def\@acknowledgmentsContent{#2} % Set content
}
% Command to set notation and acronyms page
\newcommand{\setnotationpage}[1]{\renewcommand{\@notationpage}{#1}}
\newcommand{\fichacatalografica}[1]{%
    \booltrue{hasfichacatalografica}%
    \renewcommand{\@fichapath}{#1}% Store path
}
\newcommand{\atadefesa}[1]{%
    \booltrue{hasatadefesa}%
    \renewcommand{\@atapath}{#1}% Store path
}

% \newcommand{\abstractname}{Abstract}
% \providecaptionname{brazil,portuguese,spanish}{\abstractname}{Resumo}

% Command for abstract in first language
\newcommand{\abstract}[2][\abstractname]{%
    \def\@abstractTitle{#1} % Set title
    \def\@abstractContent{#2} % Set content
}

% \newcommand{\acronymname}{Acronyms}
\newcommand{\notationname}{Notation}
\newcommand{\acknowledgmentsname}{Acknowledgments}
% checks if option noglossaries was set, if not, define new command
\ifthenelse{\equal{\@useglossaries}{true}}{%
    % Command to enable glossaries
    \RequirePackage[acronym,nonumberlist,toc=true]{glossaries}%
    % \RequirePackage{glossaries-extra}
    % \RequirePackage{glossary-mcols}
    \newglossary[nlg]{notation}{not}{ntn}{\notationname}
    \newcommand{\newnotation}[3]{%
        \newglossaryentry{#1}{%
            type=notation,
            name={#2},
            description={#3},
            sort=def
        }
    }
    % \setlength{\glsdescwidth}{0.75\hsize}

    % Allow the symbol column to wrap text
    % \setlength{\glsentrysymbolwidth}{0.2\hsize} 
    \makenoidxglossaries%
    
}{%
    % Command to disable glossaries
    \PackageInfo{macrothesis}{glossaries package not loaded. Acronym and notation management will not be available.}%
}

\newif\ifabstractsecond % Create a new conditional flag
% Command to store abstract in second language
\newcommand{\abstractsecond}[2][\abstractname]{%
    \abstractsecondtrue % Set the flag to true
    \def\@secondAbstractTitle{#1} % Set title
    \def\@secondAbstractContent{#2} % Set content
}

% Cover Page Command
\newcommand{\maketitlepage}{%
    \ifthenelse{\equal{\@ufmgformat}{true}}{%
        % This forces the plain style (enforced by book class) to be empty
        \removestyle
        \pagestyle{empty}
        \pagenumbering{arabic}
        \setcounter{page}{0}

        \begin{center}
          \textbf{\MakeUppercase{\@universityName}}
          \par
          Escola de Engenharia
          \par
          Programa de P\'{o}s-Gradua\c{c}\~{a}o em Engenharia El\'{e}trica
          \par
          
          \par
        \end{center}
          
          \vspace*{3.2cm}
          
          \begin{center}
              \@thesisAuthor\\
          \end{center}
          
          \vspace*{3.2cm}
          \begin{center}
              \textbf{\MakeUppercase{\@thesisTitle}}
          \end{center}
          \vfill
          \begin{center}
            \@locationName\\
            \the\year
        \end{center}
    }{
        \thispagestyle{empty}
        \pagenumbering{roman}
        \setcounter{page}{1}
    
      % Research group Logo
      \includegraphics[scale=0.145]{template/logo}

      % University, Department, and Research Group
      {\small \@universityName\par}
      {\small \@programName\par}
      {\small \@researchGroupName\par}

      \vspace*{6.5cm}

      % Title
      \begin{center}
          {\Large\textbf{\expandafter\MakeUppercase\expandafter{\@thesisTitle}}}\vfill{}
      \end{center}

      % Author, Location, and Year
      \begin{center}
          \textbf{\@thesisAuthor}\\
          \@locationName\\
          \the\year
      \end{center}
      }%
    \pagebreak
}

\newcommand{\preamblepage}{%
    \selectlanguage{\@mainlanguage}
    \thispagestyle{empty}

    \begin{center}
    \textbf{\@thesisAuthor}
    \end{center}

    \vfill{}

    \begin{center}
    {\Large\textbf{\expandafter\MakeUppercase\expandafter{\@thesisTitle}}}
    \end{center}

    \vfill{}
    \ifthenelse{\equal{\@masterdoc}{doctor}}{%
        \renewcommand{\titlelang}{\doclang}
    }{%
        \renewcommand{\titlelang}{\masterlang}
    }
    \ifthenelse{\equal{\@texttype}{dissertation}}{%
        \renewcommand{\typelang}{\dissertationlang}
    }{%
        \renewcommand{\typelang}{\thesislang}
    }
    \begin{flushright}
    \parbox[t]{0.6\columnwidth}{%
    {\submissiontext}
    % {\typelang} submitted to the {\@programName} of {\@universityName}, in partial fulfillment of the requirements for the degree of {\titlelang} in Electrical Engineering.

    \vspace{1cm}

    \textbf{{\advisorlang}:} \@thesisAdvisor %
    \ifx\@thesisCoadvisorA\empty
        % No coadvisor
    \else
        \ifx\@thesisCoadvisorB\empty
            \\ \textbf{{\coadvisorlang}:} \@thesisCoadvisorA% Only first coadvisor
        \else
            \\ \textbf{{\coadvisorlang}:} \@thesisCoadvisorA 
            \\ \textbf{{\coadvisorlang}:} \@thesisCoadvisorB
        \fi
    \fi
    }
    \end{flushright}

    \vspace*{\fill}

    \begin{center}
    \@locationName\\
    \the\year
    \end{center}
    \pagebreak
    % Include Ficha Catalografica
    \ifdefempty{\@fichapath}{%
        \PackageWarning{macrothesis}{%
        Cataloging Entry (Ficha Catalográfica) was not provided. Ignore this warning if this is not the final version of the text.%
        }%
    }{%
        \PackageInfo{macrothesis}{Including Cataloging Entry (Ficha Catalográfica) PDF file: '\@fichapath'}%
        \IfFileExists{\@fichapath}{%
            \cleardoublepage%
            \thispagestyle{empty}%
            \includepdf[pages=-,pagecommand={\thispagestyle{empty}}]{\@fichapath}%
            \cleardoublepage%
        }{%
            \PackageError{macrothesis}{%
            Missing Cataloging Entry (Ficha Catalográfica) PDF file at: '\@fichapath'%
            }{Check the file path and permissions}%
        }%
    }% 
    % Include Ata de Defesa (or Folha de Aprovacao)
    \ifdefempty{\@atapath}{%
        \PackageWarning{macrothesis}{%
        Defense certificate (Ata de Defesa) not provided. This warning can be ignored in draft versions.%
        }%
    }{%
        \PackageInfo{macrothesis}{Including Defense certificate (Ata de Defesa) PDF file: '\@atapath'}%
        \IfFileExists{\@atapath}{%
            \cleardoublepage%
            \thispagestyle{empty}%
            \includepdf[pages=-,pagecommand={\thispagestyle{empty}},width=\textwidth]{\@atapath}%
            \cleardoublepage%
        }{%
            \PackageError{macrothesis}{%
            Missing Defense certificate (Ata de Defesa) PDF file at: '\@atapath'%
            }{Check the file path and permissions}%
        }%
    }%    
    % Dedication page
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}

    \selectlanguage{\@secondlanguage}%
    \textbf{\thispagestyle{empty}}

    \vfill{}

    \begin{flushright}
    \begin{minipage}[t]{0.4\columnwidth}%
    \selectlanguage{\@mainlanguage}%
    \begin{singlespace}
    \begin{flushright}
    \emph{\@dedicationContent}
    \par\end{flushright}
    \end{singlespace}
    %
    \end{minipage}
    \par\end{flushright}

    \vspace{3cm}

    % Acknowledgments
    \selectlanguage{\@mainlanguage}%
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}

    \chapter*{\@acknowledgmentsTitle}
    % \thispagestyle{empty}
    \@acknowledgmentsContent

    % Second language Abstract
    \begin{onehalfspace}
    \pagebreak{}
    \end{onehalfspace}
    % \pagestyle{empty}
    % NORMA UFMG -- Primeiro Resumo (ptbr) depois Abstract (ou qualquer lingua estrangeira)
    %checks wheter second abstract is empty, if not, prints it
    \ifabstractsecond
        \selectlanguage{\@secondlanguage}%
        \chapter*{\@secondAbstractTitle}
        % \thispagestyle{empty}
        \@secondAbstractContent
    \fi
    
    \begin{singlespace}
    \pagebreak{}
    \end{singlespace}

    % First language Abstract
    \selectlanguage{\@mainlanguage}%
    \chapter*{\@abstractTitle}
    % \thispagestyle{empty}
    \@abstractContent
    
    \selectlanguage{\@mainlanguage}%
    \pagebreak{}%

    % Table of Contents, Figures and Tables
    \addtocontents{toc}{\protect{\pdfbookmark[0]{\contentsname}{toc}}}%
    \tableofcontents{}\listoffigures\listoftables
    
    \glossarypage

    \section*{\protect\pagebreak}% For some reason this is necessary

    % \pagestyle{plain}
    \ifthenelse{\equal{\@ufmgformat}{true}}{%
        % This sets the header again for plain style (enforces book class plain to follow the default style)
        \setfancyheader
    }{%
        \pagestyle{thesisstyle}
        \pagenumbering{arabic}
        \setcounter{page}{1}
    }
}

\newcommand{\glossarypage}{
    % Conditionally load glossaries
    \ifthenelse{\equal{\@useglossaries}{true}}{%
        \PackageInfo{macrothesis}{Adding all acronyms and notations whether they are used or not.}% 
        \glsaddall
        \printnoidxglossary[type=acronym,sort=word]
        \printnoidxglossary[type=notation,style=long,sort=def]
    }{%
        \selectlanguage{\@mainlanguage}%
        \@notationpage
    }
}

% ---------------------------------------------------------------------------- %
%                              LOCALE SETTINGS
% ---------------------------------------------------------------------------- %
% WARNING: Some of the translations may not be accurate. Please check them.
% Acknowledgments
\providecaptionname{brazil,portuguese}{\acknowledgmentsname}{Agradecimentos}
\providecaptionname{english,american,USenglish,canadian}{\acknowledgmentsname}{Acknowledgments}
\providecaptionname{british,australian,newzealand,UKenglish}{\acknowledgmentsname}{Acknowledgements}
\providecaptionname{spanish}{\acknowledgmentsname}{Agradecimientos}
\providecaptionname{french}{\acknowledgmentsname}{Remerciements}
\providecaptionname{german,ngerman}{\acknowledgmentsname}{Danksagungen}
% Notation
\providecaptionname{brazil,portuguese}{\notationname}{Notação}
\providecaptionname{english,american,USenglish,canadian,british,australian,
newzealand,UKenglish,french,german,ngerman}{\notationname}{Notation}
\providecaptionname{spanish}{\notationname}{Notación}
% Master
\providecaptionname{brazil,portuguese}{\masterlang}{Mestre}
\providecaptionname{english,american,USenglish,canadian,british,australian,
newzealand,UKenglish,german,ngerman}{\masterlang}{Master}
\providecaptionname{spanish}{\masterlang}{Maestro}
\providecaptionname{french}{\masterlang}{Maître}
\providecaptionname{}{\masterlang}{Master}
% Doctor
\providecaptionname{brazil,portuguese}{\doclang}{Doutor}
\providecaptionname{english,american,USenglish,canadian,british,australian,
newzealand,UKenglish,spanish}{\doclang}{Doctor}
\providecaptionname{french}{\doclang}{Docteur}
\providecaptionname{german,ngerman}{\doclang}{Doktor}
% Thesis
\providecaptionname{brazil,portuguese}{\thesislang}{Tese}
\providecaptionname{english,american,USenglish,canadian,british,australian,
newzealand,UKenglish,german,ngerman}{\thesislang}{Thesis}
\providecaptionname{spanish}{\thesislang}{Tesis}
\providecaptionname{french}{\thesislang}{Thèse}
% Dissertation
\providecaptionname{brazil,portuguese}{\dissertationlang}{Dissertação}
\providecaptionname{english,american,USenglish,canadian,british,australian,
newzealand,UKenglish,german,ngerman}{\dissertationlang}{Dissertation}
\providecaptionname{spanish}{\dissertationlang}{Disertación}
\providecaptionname{french}{\dissertationlang}{Mémoire}
% Advisor
\providecaptionname{brazil,portuguese}{\advisorlang}{Orientador}
\providecaptionname{english,american,USenglish,canadian}{\advisorlang}{Advisor}
\providecaptionname{british,australian,newzealand,UKenglish}{\advisorlang}{Supervisor}
\providecaptionname{spanish}{\advisorlang}{Director}
\providecaptionname{french}{\advisorlang}{Directeur}
\providecaptionname{german,ngerman}{\advisorlang}{Betreuer}
% Co-advisor
\providecaptionname{brazil,portuguese}{\coadvisorlang}{Coorientador}
\providecaptionname{english,american,USenglish,canadian}{\coadvisorlang}{Co-advisor}
\providecaptionname{british,australian,newzealand,UKenglish}{\coadvisorlang}{Co-supervisor}
\providecaptionname{spanish}{\coadvisorlang}{Codirector}
\providecaptionname{french}{\coadvisorlang}{Codirecteur}
\providecaptionname{german,ngerman}{\coadvisorlang}{Kobetreuer}
%Submission text
\providecaptionname{american,USenglish,english,canadian,british,australian,
newzealand,UKenglish}{\submissiontext}{{\typelang} submitted to the {\@programName} of {\@universityName}, in partial fulfillment of the requirements for the degree of {\titlelang} in {\@coursename}.}
\providecaptionname{brazil,portuguese}{\submissiontext}{{\typelang} submetida ao {\@programName} da {\@universityName}, como parte dos requisitos para obtenção do título de {\titlelang} em {\@coursename}.}
\providecaptionname{spanish}{\submissiontext}{{\typelang} presentada al {\@programName} de la {\@universityName}, como parte de los requisitos para obtener el título de {\titlelang} en {\@coursename}.}
\providecaptionname{french}{\submissiontext}{{\typelang} présentée au {\@programName} de l'{\@universityName}, en vue de l'obtention du titre de {\titlelang} en {\@coursename}.}
\providecaptionname{german,ngerman}{\submissiontext}{{\typelang} vorgelegt dem {\@programName} der {\@universityName}, als Teil der Anforderungen für den Grad {\titlelang} im Fachgebiet {\@coursename}.}


